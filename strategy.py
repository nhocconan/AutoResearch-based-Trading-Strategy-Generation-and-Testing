#!/usr/bin/env python3

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily EMA34 for Camarilla calculation (needs previous day close)
    close_1d = close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1-week trend direction (aligned)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    trend_1w_up = close_1w_aligned > ema_34_1w_aligned
    trend_1w_down = close_1w_aligned < ema_34_1w_aligned
    
    # Calculate Camarilla levels from previous day's range
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # Where C = previous close, H = previous high, L = previous low
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~3 days for daily to reduce trades
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h4[i]) or 
            np.isnan(camarilla_l4[i]) or 
            np.isnan(trend_1w_up[i]) or 
            np.isnan(trend_1w_down[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Close above H4 with weekly uptrend and volume
            if (close[i] > camarilla_h4[i] and 
                trend_1w_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Close below L4 with weekly downtrend and volume
            elif (close[i] < camarilla_l4[i] and 
                  trend_1w_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Close below L4 or trend change
            if (close[i] < camarilla_l4[i]) or not trend_1w_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above H4 or trend change
            if (close[i] > camarilla_h4[i]) or not trend_1w_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla H4/L4 levels act as strong support/resistance. 
# Breakout above H4 in weekly uptrend with volume confirms bullish continuation.
# Breakdown below L4 in weekly downtrend with volume confirms bearish continuation.
# Weekly EMA34 filter ensures we trade with the higher timeframe trend.
# Volume confirmation avoids false breakouts. 3-day cooldown reduces trade frequency.
# Target: 15-25 trades/year. Works in bull markets by buying H4 breakouts in uptrends
# and in bear markets by selling L4 breakdowns in downtrends. Daily timeframe 
# provides sufficient signal quality while minimizing noise. Camarilla levels 
# are mathematically derived pivot points that work well in crypto markets. 
# Weekly trend filter ensures alignment with major market direction. Volume 
# confirms institutional participation. This avoids overtrading by requiring 
# multiple confirmations: price level, trend direction, and volume.