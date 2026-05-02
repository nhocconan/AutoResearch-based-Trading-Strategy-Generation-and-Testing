#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation
# Camarilla pivot levels provide precise intraday support/resistance; 1d EMA(34) filters primary trend
# Volume spike (2.0x 20-period average) ensures strong participation and reduces false breakouts
# Uses discrete position sizing 0.25 to minimize fee churn
# Targets 12-25 trades/year (50-100 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by requiring volume confirmation and primary trend alignment

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 12h Camarilla levels (R3, S3) from prior 12h bar
    # Camarilla formulas: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # We use prior bar's OHLC to avoid look-ahead
    prior_close = np.roll(close, 1)
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_close[0] = np.nan  # first bar has no prior
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    
    camarilla_range = prior_high - prior_low
    camarilla_factor = 1.1 * 1.1 / 4  # 1.1 * 1.1 / 4 = 0.3025
    camarilla_upper = prior_close + camarilla_range * camarilla_factor  # R3
    camarilla_lower = prior_close - camarilla_range * camarilla_factor  # S3
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for volume MA and prior bar data)
    start_idx = 20  # 20 for volume MA
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_upper[i]) or np.isnan(camarilla_lower[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + price > 1d EMA + volume spike
            if close[i] > camarilla_upper[i] and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + price < 1d EMA + volume spike
            elif close[i] < camarilla_lower[i] and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retreats to Camarilla H3/L3 level (midpoint)
            camarilla_mid = prior_close[i]  # Camarilla H3/L3 is essentially the prior close
            if close[i] < camarilla_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises to Camarilla H3/L3 level (midpoint)
            camarilla_mid = prior_close[i]  # Camarilla H3/L3 is essentially the prior close
            if close[i] > camarilla_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals