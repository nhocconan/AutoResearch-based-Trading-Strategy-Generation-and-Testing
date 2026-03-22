#!/usr/bin/env python3
"""
Experiment #026: 30m Fisher Transform + 4h HMA Trend + Vol Regime Filter
Hypothesis: Ehlers Fisher Transform (period=9) excels at catching reversals in bear/range markets.
Combined with 4h HMA trend bias (not hard filter) to reduce counter-trend trades.
Volatility regime (ATR ratio) determines entry aggressiveness: high vol = mean reversion, low vol = trend follow.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Key innovation: Fisher Transform normalizes price to Gaussian distribution, making extremes statistically significant.
Entry: Fisher crosses -1.5 (long) or +1.5 (short) with optional trend/vol confirmation.
Stoploss: 2.5*ATR trailing stop to survive 2022-style crashes.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_fisher_4h_hma_vol_regime_v1"
timeframe = "30m"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X is normalized price.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5.
    Research shows superior reversal detection in bear/range markets.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    # Calculate typical price and normalize
    for i in range(period, n):
        # Use (high + low) / 2 as price input
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize to -1 to +1 range
        if highest == lowest:
            continue
        
        x = (hl2 - lowest) / (highest - lowest)
        x = 0.999 * (2.0 * x - 1.0)  # Scale to -0.999 to +0.999
        
        # Fisher transform
        if x >= 1.0 or x <= -1.0:
            continue
            
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Signal line (1-period lag)
        if i > period:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

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
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / (std + 1e-10)
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
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    # ATR ratio for volatility regime
    atr_ratio = atr_7 / (atr_30 + 1e-10)
    atr_ratio[np.isnan(atr_ratio)] = 1.0
    
    # Fisher Transform
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Z-score for mean reversion
    zscore = calculate_zscore(close, period=20)
    
    # EMA for additional trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (soft filter, not hard requirement)
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # Volatility regime
        high_vol = atr_ratio[i] > 1.5  # Vol spike
        low_vol = atr_ratio[i] < 0.8   # Vol compression
        
        # Fisher Transform signals
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        
        # Price position vs Bollinger Bands
        price_below_lower = close[i] < bb_lower[i]
        price_above_upper = close[i] > bb_upper[i]
        
        # Z-score extremes
        zscore_oversold = zscore[i] < -2.0
        zscore_overbought = zscore[i] > 2.0
        
        # EMA trend
        ema_bullish = close[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Fisher cross up from extreme + bull trend
        if fisher_cross_up and bull_trend:
            new_signal = SIZE_MAX
        # Secondary: Fisher extreme low + price below BB + any trend
        elif fisher_extreme_low and price_below_lower:
            new_signal = SIZE_BASE
        # Tertiary: Fisher cross up + z-score oversold (mean reversion)
        elif fisher_cross_up and zscore_oversold:
            new_signal = SIZE_BASE
        # Quaternary: Fisher cross up + high vol (vol spike reversion)
        elif fisher_cross_up and high_vol:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: Fisher cross down from extreme + bear trend
        if fisher_cross_down and bear_trend:
            new_signal = -SIZE_MAX
        # Secondary: Fisher extreme high + price above BB + any trend
        elif fisher_extreme_high and price_above_upper:
            new_signal = -SIZE_BASE
        # Tertiary: Fisher cross down + z-score overbought (mean reversion)
        elif fisher_cross_down and zscore_overbought:
            new_signal = -SIZE_BASE
        # Quaternary: Fisher cross down + high vol (vol spike reversion)
        elif fisher_cross_down and high_vol:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for wider stops)
            current_stop = highest_close - 2.5 * atr_14[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr_14[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr_14[i] if position_side > 0 else close[i] + 2.5 * atr_14[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals