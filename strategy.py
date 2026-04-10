#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and 1d volume confirmation
# - Uses 4h EMA(21) for trend direction (long when price > EMA, short when price < EMA)
# - Uses 1d volume > 1.5x 20-period volume SMA for confirmation
# - Enters on 1h Camarilla H3/L3 breakout in direction of 4h trend
# - Exits at Camarilla H4/L4 levels or opposite H3/L3 breakout
# - Position sizing: 0.20 discrete level to minimize fee impact
# - Session filter: 08-20 UTC to avoid low-volume periods
# - Target: 15-35 trades/year on 1h timeframe (~60-140 over 4 years)

name = "1h_4h_1d_camarilla_trend_volume_v1"
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
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h EMA(21) for trend filter
    ema_period = 21
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 1h Camarilla pivot points (using previous bar's OHLC)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Camarilla levels based on previous bar
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        range_ = prev_high - prev_low
        
        camarilla_h3[i] = prev_close + range_ * 1.1 / 4
        camarilla_l3[i] = prev_close - range_ * 1.1 / 4
        camarilla_h4[i] = prev_close + range_ * 1.1 / 2
        camarilla_l4[i] = prev_close - range_ * 1.1 / 2
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume_1d[i // 6] > 1.5 * volume_sma_20_1d_aligned[i] if (i // 6) < len(volume_1d) else False
        
        # Trend filter: price vs 4h EMA
        trend_long = close[i] > ema_4h_aligned[i]
        trend_short = close[i] < ema_4h_aligned[i]
        
        # Camarilla breakout conditions
        breakout_h3 = close[i] > camarilla_h3[i]  # Break above H3
        breakdown_l3 = close[i] < camarilla_l3[i]  # Break below L3
        
        if position == 0:  # Flat - look for entry
            if in_session and vol_confirm:
                if trend_long and breakout_h3:
                    position = 1
                    signals[i] = 0.20
                elif trend_short and breakdown_l3:
                    position = -1
                    signals[i] = -0.20
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit at H4 level or opposite L3 breakdown
            exit_long = close[i] >= camarilla_h4[i] or breakdown_l3
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        else:  # position == -1 (Short position) - look for exit
            # Exit at L4 level or opposite H3 breakout
            exit_short = close[i] <= camarilla_l4[i] or breakout_h3
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
    
    return signals