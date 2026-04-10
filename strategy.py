#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume confirmation and 1d trend filter
# - Primary: 1h price breaks above/below Camarilla H3/L3 levels (strong intraday support/resistance)
# - Volume filter: 4h volume > 1.2x 20-period volume MA to confirm breakout with participation
# - Trend filter: 1d EMA(50) slope > 0 (for longs) or < 0 (for shorts) to align with daily momentum
# - Session filter: Trade only between 08:00-20:00 UTC to avoid low-liquidity periods
# - Exit: Price returns to Camarilla H4/L4 levels or breaks opposite Camarilla level
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Works in bull/bear: Camarilla levels adapt to volatility, volume confirms institutional interest,
#   EMA filter ensures we trade with daily trend, session filter reduces noise

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla pivot levels for 1h timeframe (based on previous bar)
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
    
    # Calculate 4h volume confirmation: volume > 1.2x 20-period volume MA
    volume_ma_20_4h = pd.Series(volume_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_slope_1d = np.diff(ema_50_1d_aligned, prepend=ema_50_1d_aligned[0])
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(volume_ma_20_4h_aligned[i]) or np.isnan(ema_50_slope_1d[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 4h volume > 1.2x 20-period MA
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)
        vol_confirm = vol_4h_current[i] > 1.2 * volume_ma_20_4h_aligned[i]
        
        if position == 0:  # Flat - look for new Camarilla breakouts
            # Long entry: Price breaks above H3 + vol confirmation + daily uptrend
            if close[i] > camarilla_h3[i] and vol_confirm and ema_50_slope_1d[i] > 0:
                position = 1
                signals[i] = 0.20
            # Short entry: Price breaks below L3 + vol confirmation + daily downtrend
            elif close[i] < camarilla_l3[i] and vol_confirm and ema_50_slope_1d[i] < 0:
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to H4/L4 levels or breaks opposite Camarilla level
            if position == 1:  # Long position
                if close[i] <= camarilla_h4[i] or close[i] < camarilla_l3[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if close[i] >= camarilla_l4[i] or close[i] > camarilla_h3[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals