#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide precise intraday support/resistance for breakout entries
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades
# Volume confirmation (>1.5x 20-period EMA) filters for institutional participation
# Designed for 4h timeframe targeting 19-50 trades/year (75-200 total over 4 years)
# Works in bull markets (price > daily EMA34 + break above R3) and bear markets (price < daily EMA34 + break below S3)
# Uses discrete position sizing (0.30) to balance return potential with drawdown control

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical Price = (High + Low + Close) / 3
    # Range = High - Low
    # R3 = Close + (Range * 1.1/2) = Close + (Range * 0.55)
    # S3 = Close - (Range * 1.1/2) = Close - (Range * 0.55)
    typical_price = (high + low + close) / 3
    price_range = high - low
    camarilla_r3 = typical_price + (price_range * 0.55)
    camarilla_s3 = typical_price - (price_range * 0.55)
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias and close[i] > camarilla_r3[i] and volume_confirmation[i]:
                # Long: Daily trend up, price breaks above R3, volume confirmation
                signals[i] = 0.30
                position = 1
            elif bearish_bias and close[i] < camarilla_s3[i] and volume_confirmation[i]:
                # Short: Daily trend down, price breaks below S3, volume confirmation
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Daily trend turns bearish OR price falls below S3 (reversal signal)
            if (not bullish_bias) or (close[i] < camarilla_s3[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Daily trend turns bullish OR price rises above R3 (reversal signal)
            if (not bearish_bias) or (close[i] > camarilla_r3[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals