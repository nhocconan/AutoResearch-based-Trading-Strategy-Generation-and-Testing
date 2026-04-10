#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume spike
# - Long when price breaks above Camarilla H3 level AND 4h close > 4h EMA50 (uptrend) AND volume > 1.5x 20-period average
# - Short when price breaks below Camarilla L3 level AND 4h close < 4h EMA50 (downtrend) AND volume > 1.5x 20-period average
# - Exit when price returns to Camarilla PIVOT level (mean reversion to equilibrium)
# - Uses discrete position sizing 0.20 to limit fee churn
# - Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years)
# - Session filter: only trade 08-20 UTC to avoid low-volume Asian session
# - Uses 4h for trend direction, 1h for precise entry timing to reduce false breakouts

name = "1h_4h_camarilla_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Pre-compute 1h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 1h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Pre-compute 1h Camarilla levels from previous period's OHLC
    # Camarilla levels use previous period's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current values
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    camarilla_h3 = pivot + (range_hl * 1.1 / 4)
    camarilla_l3 = pivot - (range_hl * 1.1 / 4)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(pivot[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_4h_aligned[i]) or not session_filter[i]):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND 4h uptrend AND volume spike
            if (close[i] > camarilla_h3[i] and 
                close[i] > ema_4h_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.20
            # Short conditions: price breaks below Camarilla L3 AND 4h downtrend AND volume spike
            elif (close[i] < camarilla_l3[i] and 
                  close[i] < ema_4h_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to pivot (mean reversion)
            # Exit when price returns to pivot level (mean reversion to equilibrium)
            exit_long = (position == 1 and close[i] <= pivot[i])
            exit_short = (position == -1 and close[i] >= pivot[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.20
                else:
                    signals[i] = -0.20
    
    return signals