#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d: breakout continuation at R4/S4 with volume confirmation and ATR filter
# - Long: price breaks above R4 with volume > 1.5x ATR-scaled average and ATR(14) > 0.01*close (volatility filter)
# - Short: price breaks below S4 with same volume and ATR conditions
# - Exit: mean reversion at R3/S3 levels or ATR-based stoploss (2*ATR from entry)
# - Uses 1d Camarilla levels from prior day OHLC, aligned to 4h
# - Works in bull markets (breakout continuation) and bear markets (fade extremes at R4/S4 then revert to R3/S3)
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Discrete position sizing: 0.25 magnitude to balance risk and reward

name = "4h_1d_camarilla_breakout_atr_v1"
timeframe = "4h"
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
    entry_price = 0.0  # track entry price for stoploss
    
    # Load 1d data ONCE before loop for Camarilla levels (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Pre-compute 1d Camarilla levels (based on prior day OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.1*(high-low)*1.1/2
    # R3 = close + 1.1*(high-low)*1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4
    # S4 = close - 1.1*(high-low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_r3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * range_1d * 1.1 / 4
    camarilla_s4 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels for current day)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute ATR(14) for volatility filter and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: volume > 1.5 * ATR-scaled average volume
    # Use ATR to normalize volume threshold across different volatility regimes
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma_20  # base threshold
    # Adjust threshold by ATR to make it volatility-sensitive: higher ATR = higher volume needed
    atr_ratio = atr / close  # ATR as fraction of price
    atr_ratio_ma = pd.Series(atr_ratio).rolling(window=20, min_periods=1).mean().values  # smooth
    vol_threshold_adj = vol_threshold * (1 + atr_ratio_ma)  # increase threshold in high volatility
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_threshold_adj[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > volatility-adjusted threshold
        vol_confirm = volume_current > vol_threshold_adj[i]
        
        # ATR filter: only trade when volatility is reasonable (avoid extremely low/high vol)
        atr_filter = (atr[i] > 0.005 * close_price) & (atr[i] < 0.05 * close_price)
        
        # Price position relative to Camarilla levels
        r4 = camarilla_r4_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        s4 = camarilla_s4_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above R4 with volume confirmation and ATR filter
        if close_price > r4 and vol_confirm and atr_filter:
            enter_long = True
        
        # Short breakout: price breaks below S4 with volume confirmation and ATR filter
        if close_price < s4 and vol_confirm and atr_filter:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: mean reversion at R3 OR stoploss (2*ATR below entry)
            exit_long = close_price <= r3 or close_price < entry_price - 2.0 * atr[i]
        elif position == -1:
            # Exit short: mean reversion at S3 OR stoploss (2*ATR above entry)
            exit_short = close_price >= s3 or close_price > entry_price + 2.0 * atr[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals