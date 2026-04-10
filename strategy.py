#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume confirmation and 1w trend filter
# - Primary: 4h price breaks above/below Camarilla H3/L3 levels (strong intraday support/resistance)
# - Volume filter: 12h volume > 1.2x 24-period volume MA to confirm breakout with participation
# - Trend filter: 1w EMA(50) slope > 0 (for longs) or < 0 (for shorts) to align with weekly momentum
# - Exit: Price returns to Camarilla H4/L4 levels or opposite Camarilla breakout
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla levels adapt to volatility, volume confirms institutional interest,
#   EMA filter ensures we trade with weekly trend, effective in both bull and bear markets

name = "4h_12h_1w_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    if len(df_12h) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla pivot levels for 4h timeframe (based on previous bar)
    # Camarilla: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4
    #            L3 = close - 1.1*(high-low)/4, L4 = close - 1.1*(high-low)/2
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_h3 = prev_close + 1.1 * camarilla_range / 4
    camarilla_l3 = prev_close - 1.1 * camarilla_range / 4
    camarilla_h4 = prev_close + 1.1 * camarilla_range / 2
    camarilla_l4 = prev_close - 1.1 * camarilla_range / 2
    
    # Calculate 12h volume confirmation: volume > 1.2x 24-period volume MA
    volume_ma_24_12h = pd.Series(volume_12h).ewm(span=24, min_periods=24, adjust=False).mean().values
    volume_ma_24_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_24_12h)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_50_slope_1w = np.diff(ema_50_1w_aligned, prepend=ema_50_1w_aligned[0])
    
    for i in range(24, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(volume_ma_24_12h_aligned[i]) or np.isnan(ema_50_slope_1w[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 12h volume > 1.2x 24-period MA
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)
        vol_confirm = vol_12h_current[i] > 1.2 * volume_ma_24_12h_aligned[i]
        
        if position == 0:  # Flat - look for new Camarilla breakouts
            # Long entry: Price breaks above H3 + vol confirmation + weekly uptrend
            if close[i] > camarilla_h3[i] and vol_confirm and ema_50_slope_1w[i] > 0:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below L3 + vol confirmation + weekly downtrend
            elif close[i] < camarilla_l3[i] and vol_confirm and ema_50_slope_1w[i] < 0:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to H4/L4 levels or breaks opposite Camarilla level
            if position == 1:  # Long position
                if close[i] <= camarilla_h4[i] or close[i] < camarilla_l3[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= camarilla_l4[i] or close[i] > camarilla_h3[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals