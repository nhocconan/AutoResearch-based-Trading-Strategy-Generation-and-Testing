#!/usr/bin/env python3
"""
Experiment #221: 12h Fisher Transform + BB Mean Reversion + 1d HMA Trend Filter

Hypothesis: 12h timeframe with Ehlers Fisher Transform catches reversals better than 
RSI/MACD in bear/range markets (2025 test period). Fisher Transform normalizes price 
to Gaussian distribution, making extremes (-2 to +2) reliable reversal signals. 
Combined with Bollinger Band mean reversion (price at BB extremes) and 1d HMA trend 
bias, this should work in both trending and ranging conditions.

Why this might work when others failed:
- Fisher Transform is designed for reversal detection (not trend following)
- BB extremes provide concrete mean reversion levels
- 1d HMA filter prevents counter-trend trades in strong trends
- 12h bars = 2 per day, fewer trades = less fee drag
- Conservative sizing (0.25) controls drawdown

Key difference from failed strategies:
- #209 (regime adaptive): Too complex, regime detection lagged
- #215 (KAMA): Pure trend following fails in ranges
- This uses MEAN REVERSION with trend filter (hybrid approach)

Timeframe: 12h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_bb_1d_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Values typically range from -2 to +2. Extremes indicate reversals.
    Long when Fisher crosses above -1.5 from below.
    Short when Fisher crosses below +1.5 from above.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_prev = np.zeros(n)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        normalized = (typical[i] - lowest) / range_val
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher Transform formula
        fisher_raw = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Smooth with EMA
        if i == period:
            fisher[i] = fisher_raw
            fisher_prev[i] = fisher_raw
        else:
            fisher[i] = 0.7 * fisher_raw + 0.3 * fisher[i-1]
            fisher_prev[i] = fisher[i-1] if i > 0 else fisher_raw
    
    return fisher, fisher_prev

def calculate_bollinger_bands(close, period=20, std_dev=2.5):
    """Calculate Bollinger Bands with wider std_dev for mean reversion."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.5)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(bb_upper[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_long_signal = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_short_signal = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === BOLLINGER BAND MEAN REVERSION ===
        # Price at lower BB = oversold (long opportunity)
        # Price at upper BB = overbought (short opportunity)
        at_bb_lower = close[i] <= bb_lower[i]
        at_bb_upper = close[i] >= bb_upper[i]
        
        # === RSI EXTREMES ===
        # RSI < 30 = oversold
        # RSI > 70 = overbought
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # === ADX FILTER ===
        # ADX < 25 = ranging market (favor mean reversion)
        # ADX > 25 = trending market (favor trend following)
        is_ranging = adx[i] < 25
        is_trending = adx[i] >= 25
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long entries (multiple paths to ensure enough trades):
        # Path 1: Fisher reversal + BB lower + 1d bullish bias
        # Path 2: Fisher reversal + RSI oversold + 1d bullish bias
        # Path 3: BB lower + RSI oversold + ranging market (mean reversion)
        
        if fisher_long_signal:
            # Fisher reversal with trend bias
            if bull_trend_1d and (at_bb_lower or rsi_oversold):
                new_signal = SIZE_BASE
            # Fisher reversal in ranging market (no trend bias needed)
            elif is_ranging and (at_bb_lower or rsi_oversold):
                new_signal = SIZE_BASE
        
        # Also allow long at BB lower even without Fisher if conditions align
        if at_bb_lower and rsi_oversold:
            if bull_trend_1d or is_ranging:
                if new_signal == 0.0:  # Don't override if already set
                    new_signal = SIZE_BASE
        
        # Short entries (mirror logic):
        # Path 1: Fisher reversal + BB upper + 1d bearish bias
        # Path 2: Fisher reversal + RSI overbought + 1d bearish bias
        # Path 3: BB upper + RSI overbought + ranging market
        
        if fisher_short_signal:
            # Fisher reversal with trend bias
            if bear_trend_1d and (at_bb_upper or rsi_overbought):
                new_signal = -SIZE_BASE
            # Fisher reversal in ranging market
            elif is_ranging and (at_bb_upper or rsi_overbought):
                new_signal = -SIZE_BASE
        
        # Also allow short at BB upper even without Fisher
        if at_bb_upper and rsi_overbought:
            if bear_trend_1d or is_ranging:
                if new_signal == 0.0:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and new_signal != 0.0:
            # Keep existing position if stoploss not hit
            pass
        elif in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction, update extremes
            elif position_side > 0 and close[i] > highest_close:
                highest_close = close[i]
            elif position_side < 0 and (lowest_close == 0.0 or close[i] < lowest_close):
                lowest_close = close[i]
        else:
            # Exiting position
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals