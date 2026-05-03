#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# In bull regime (price > 1d EMA34), go long on break above R3 with volume spike.
# In bear regime (price < 1d EMA34), go short on break below S3 with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn and ATR-based stoploss via signal=0.
# Designed for lower trade frequency (~12-37/year) to avoid fee drag on 12h timeframe.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 12h Camarilla levels (R3, S3) using prior 12h bar's OHLC
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # We use the prior completed 12h bar to avoid look-ahead
    prior_close = np.roll(close, 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close[0] = close[0]  # avoid NaN on first bar
    prior_high[0] = high[0]
    prior_low[0] = low[0]
    
    camarilla_range = prior_high - prior_low
    r3 = prior_close + 1.1 * camarilla_range / 2
    s3 = prior_close - 1.1 * camarilla_range / 2
    
    # Calculate volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        vol_spike = volume_spike[i]
        ema_trend = ema_34_aligned[i]
        r3_val = r3[i]
        s3_val = s3[i]
        
        # Skip if any value is NaN
        if np.isnan(close_val) or np.isnan(ema_trend) or np.isnan(r3_val) or np.isnan(s3_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine regime: bull if close > 1d EMA34, bear if close < 1d EMA34
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Regime-based entry conditions
        if is_bull_regime:
            # Long: break above R3 with volume spike
            long_entry = (close_val > r3_val) and vol_spike
        else:
            long_entry = False
            
        if is_bear_regime:
            # Short: break below S3 with volume spike
            short_entry = (close_val < s3_val) and vol_spike
        else:
            short_entry = False
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on close below R3 (breakdown) or regime change to bear
            if close_val < r3_val or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on close above S3 (breakout) or regime change to bull
            if close_val > s3_val or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals