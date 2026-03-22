#!/usr/bin/env python3
"""
Experiment #022: 4h Volatility Compression + HTF Trend with ATR Stoploss

Hypothesis: After 21 failed experiments, return to proven volatility-based entries
with proper HTF confirmation. This strategy combines:

1. 4H ATR Ratio (ATR7/ATR30) - detects volatility compression (<0.7) or expansion (>1.5)
2. 1D HMA(21) - major trend direction filter (only trade with HTF trend)
3. 4H Bollinger Band position - enter on mean reversion within trend
4. 4H RSI(14) - avoid overbought/oversold extremes for entry timing
5. ATR(14) trailing stop - 2.5 ATR exit to protect capital

Why this should work when others failed:
- Volatility compression precedes breakouts (proven in literature)
- 4h timeframe = natural 30-50 trades/year (fee-efficient)
- 1d HMA filter = avoids counter-trend trades in strong trends
- BB position + RSI = precise entry timing within volatility regime
- Discrete sizing (0.25/0.30) = minimizes fee churn

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_compress_1d_hma_bb_rsi_atr_v1"
timeframe = "4h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_mult * std)
    lower = sma - (std_mult * std)
    
    return upper.values, lower.values, sma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_atr_ratio(atr_short, atr_long):
    """Calculate ATR ratio for volatility regime detection."""
    ratio = atr_short / atr_long
    ratio = np.where(atr_long == 0, 0.0, ratio)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    
    # ATR ratio for volatility regime
    atr_ratio = calculate_atr_ratio(atr_7, atr_30)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_LONG = 0.28
    BASE_SIZE_SHORT = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(atr_ratio[i]) or atr_ratio[i] == 0:
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1D MAJOR TREND DIRECTION ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === VOLATILITY REGIME ===
        vol_compressed = atr_ratio[i] < 0.75  # Low vol = potential breakout
        vol_expanded = atr_ratio[i] > 1.5  # High vol = potential reversal
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 0 else 0.5
        bb_lower_half = bb_position < 0.4  # Price in lower half of BB
        bb_upper_half = bb_position > 0.6  # Price in upper half of BB
        bb_near_lower = close[i] < bb_lower[i] * 1.02  # Near or below lower band
        bb_near_upper = close[i] > bb_upper[i] * 0.98  # Near or above upper band
        
        # === RSI FILTER ===
        rsi_neutral = 35 < rsi_14[i] < 65  # Not extreme
        rsi_oversold = rsi_14[i] < 45
        rsi_overbought = rsi_14[i] > 55
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Trend up + vol compression + BB lower + RSI not overbought
        long_score = 0
        if daily_bullish:
            long_score += 2  # Strong weight on HTF trend
        if vol_compressed:
            long_score += 1
        if bb_lower_half or bb_near_lower:
            long_score += 1
        if rsi_oversold or rsi_neutral:
            long_score += 1
        
        # Need score >= 4 for long entry (strong confluence)
        if long_score >= 4 and daily_bullish:
            new_signal = BASE_SIZE_LONG
        
        # SHORT ENTRY: Trend down + vol compression + BB upper + RSI not oversold
        short_score = 0
        if daily_bearish:
            short_score += 2  # Strong weight on HTF trend
        if vol_compressed:
            short_score += 1
        if bb_upper_half or bb_near_upper:
            short_score += 1
        if rsi_overbought or rsi_neutral:
            short_score += 1
        
        # Need score >= 4 for short entry (strong confluence)
        if short_score >= 4 and daily_bearish:
            new_signal = -BASE_SIZE_SHORT
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d trend turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if 1d trend turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals