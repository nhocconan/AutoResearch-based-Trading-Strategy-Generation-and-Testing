#!/usr/bin/env python3
"""
Experiment #242: 30m Fisher Transform + Vol Spike Mean Reversion + 4h Trend Filter

Hypothesis: In bear/range markets (2025), pure trend-following fails but reversal 
strategies excel. This combines:
1. Ehlers Fisher Transform (period=9) - catches reversals with less lag than RSI
2. Volatility Spike Mean Reversion - ATR(7)/ATR(30) > 2.0 indicates panic, revert when vol crushes
3. 4h HMA trend filter - only take longs when 4h trend bullish, shorts when bearish
4. Bollinger Band confirmation - price must be at extremes for entry

Why this might work:
- Fisher Transform normalizes price to Gaussian distribution, better reversal signals
- Vol spike + mean reversion captures "panic bottom" and "euphoria top" patterns
- 4h filter prevents counter-trend trades in strong moves
- 30m timeframe balances signal quality vs trade frequency (more trades than 4h)
- Conservative sizing (0.25) with 2.5*ATR stoploss controls drawdown

Learning from failures:
- #236 (30m Fisher + KAMA): Sharpe=-9.213 - no vol filter, wrong exit logic
- #235 (15m trend pullback): Sharpe=-3.648 - too noisy, trend-only
- #241 (15m trend pullback): Sharpe=-3.471 - same issues
- Pure trend strategies fail in 2025 bear market
- Need mean reversion + regime filter

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_vol_spike_4h_hma_bb_atr_v1"
timeframe = "30m"
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
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian-like distribution for better reversal detection.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * ((price - LL) / (HH - LL) - 0.5)
    """
    n = len(close)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        if hh > ll:
            # Normalize price to 0-1 range
            x = 0.67 * ((close[i] - ll) / (hh - ll) - 0.5)
            # Clamp to avoid division by zero
            x = np.clip(x, -0.999, 0.999)
            # Fisher transform
            fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
            # Trigger line (previous Fisher value)
            trigger[i] = fisher[i - 1] if i > 0 else 0.0
        else:
            fisher[i] = fisher[i - 1] if i > 0 else 0.0
            trigger[i] = trigger[i - 1] if i > 0 else 0.0
    
    return fisher, trigger

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # Bandwidth
    
    return upper.values, lower.values, sma.values, width

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

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    fisher, fisher_trigger = calculate_fisher_transform(high, low, close, 9)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    zscore_20 = calculate_zscore(close, 20)
    hma_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY REGIME ===
        # ATR(7)/ATR(30) > 2.0 = volatility spike (panic/euphoria)
        # ATR(7)/ATR(30) < 1.2 = volatility crush (calm after storm)
        vol_ratio = atr_7[i] / (atr_30[i] + 1e-10) if not np.isnan(atr_30[i]) else 1.0
        vol_spike = vol_ratio > 2.0
        vol_crush = vol_ratio < 1.2
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_long_cross = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        fisher_short_cross = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # === BOLLINGER BAND SIGNALS ===
        price_at_bb_lower = close[i] < bb_lower[i]
        price_at_bb_upper = close[i] > bb_upper[i]
        
        # === Z-SCORE SIGNALS ===
        zscore_extreme_low = zscore_20[i] < -2.0
        zscore_extreme_high = zscore_20[i] > 2.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Must have: 4h bullish bias + (Fisher reversal OR vol spike mean reversion)
        if bull_trend_4h:
            # Fisher reversal long
            if fisher_long_cross and rsi_oversold:
                new_signal = SIZE_BASE
            # Vol spike mean reversion long
            elif vol_spike and price_at_bb_lower and zscore_extreme_low:
                new_signal = SIZE_BASE
            # BB mean reversion in calm market
            elif vol_crush and price_at_bb_lower and fisher_oversold:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Must have: 4h bearish bias + (Fisher reversal OR vol spike mean reversion)
        if bear_trend_4h:
            # Fisher reversal short
            if fisher_short_cross and rsi_overbought:
                new_signal = -SIZE_BASE
            # Vol spike mean reversion short
            elif vol_spike and price_at_bb_upper and zscore_extreme_high:
                new_signal = -SIZE_BASE
            # BB mean reversion in calm market
            elif vol_crush and price_at_bb_upper and fisher_overbought:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals