#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction with 1w ADX filter and volume confirmation
# - Uses Kaufman Adaptive Moving Average (KAMA) to identify adaptive trend
# - Long when price > KAMA(10,2,30) and 1w ADX > 25 and volume > 1.5x 20-period average
# - Short when price < KAMA(10,2,30) and 1w ADX > 25 and volume > 1.5x 20-period average
# - Exit when price crosses back over KAMA or ATR stoploss hit
# - KAMA adapts to market noise, reducing whipsaw in ranging markets
# - 1w ADX ensures we only trade in strong weekly trends
# - Volume confirmation ensures breakouts have conviction
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits

name = "1d_1w_kama_adx_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 1w data ONCE before loop for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    tr_14 = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute KAMA(10,2,30) on 1d timeframe
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = np.abs(close_s - close_s.shift(10))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=10, min_periods=10).sum()
    er = change / volatility
    er = er.fillna(0).values
    
    # Smoothing Constants
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Pre-compute ATR for stoploss (1d timeframe)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(atr_14[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        volume_current = volume[i]
        kama_value = kama[i]
        atr_value = atr_14[i]
        adx_value = adx_aligned[i]
        vol_avg = volume_sma_20[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * vol_avg
        
        # Trend filter: 1w ADX > 25 (indicates trending market)
        adx_trend = adx_value > 25
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long when price above KAMA, volume confirmation, trending
        if close_price > kama_value and vol_confirm and adx_trend:
            enter_long = True
        
        # Short when price below KAMA, volume confirmation, trending
        if close_price < kama_value and vol_confirm and adx_trend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price crosses below KAMA or ATR-based stop
            exit_long = (close_price <= kama_value) or (close_price <= entry_price - 2.5 * atr_value)
        elif position == -1:
            # Exit short if price crosses above KAMA or ATR-based stop
            exit_short = (close_price >= kama_value) or (close_price >= entry_price + 2.5 * atr_value)
        
        # Track entry price for stoploss calculation
        if enter_long or enter_short:
            entry_price = close_price
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals