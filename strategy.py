#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses ATR(24) trailing stop for risk management. Discrete sizing 0.25 to balance return and fee drag.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakouts, in bear via short signals.
# Proven pattern from top performers: price channel + HTF trend + volume confirmation + ATR stop.
# Focus on 6h timeframe to reduce trade frequency vs lower timeframes while maintaining sufficient sample size.

name = "6h_Camarilla_R3_S3_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Camarilla levels from prior 6h bar (using close of previous completed 6h bar)
    # We need to get 6h OHLC data, but since we're on 6h timeframe, we can use the current bar's data
    # Camarilla levels are calculated based on previous day's range, but for intraday we use previous bar
    # For 6h timeframe, we'll use the prior completed 6h bar's high/low/close
    
    # Calculate typical price for prior bar
    typical_price = (high + low + close) / 3
    # Shift by 1 to get prior bar's values
    prior_typical = np.roll(typical_price, 1)
    prior_typical[0] = np.nan  # First value has no prior
    
    prior_high = np.roll(high, 1)
    prior_high[0] = np.nan
    prior_low = np.roll(low, 1)
    prior_low[0] = np.nan
    prior_close = np.roll(close, 1)
    prior_close[0] = np.nan
    
    # Camarilla levels calculation (based on prior bar's range)
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    # We'll use R3/S3 for fade/breakout logic
    
    # Calculate using prior completed bar to avoid look-ahead
    prior_range = prior_high - prior_low
    camarilla_r3 = prior_close + 1.1 * prior_range
    camarilla_s3 = prior_close - 1.1 * prior_range
    
    # Calculate 1d EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(24) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, min_periods=24, adjust=False).mean().values
    
    # Volume confirmation: volume > 2.0x 24-bar average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        camarilla_r3_val = camarilla_r3[i]
        camarilla_s3_val = camarilla_s3[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        
        # Skip if any value is NaN
        if np.isnan(camarilla_r3_val) or np.isnan(camarilla_s3_val) or np.isnan(ema_trend) or np.isnan(atr_val):
            continue
            
        # Entry conditions
        # Long: break above R3 with volume spike and above 1d EMA34
        long_entry = (close[i] > camarilla_r3_val) and (close[i] > ema_trend) and vol_spike
        # Short: break below S3 with volume spike and below 1d EMA34
        short_entry = (close[i] < camarilla_s3_val) and (close[i] < ema_trend) and vol_spike
        
        # Exit conditions (trailing stop)
        long_exit = False
        short_exit = False
        
        if position == 1:  # Long position
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            long_exit = close[i] < (highest_high_since_entry - 2.5 * atr_val)
        elif position == -1:  # Short position
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            short_exit = close[i] > (lowest_low_since_entry + 2.5 * atr_val)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals