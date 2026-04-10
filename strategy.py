#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and 1w trend filter
# - Primary: 12h price breaks above Camarilla H3 (long) or below L3 (short) from prior 1d session
# - HTF volume filter: 1d volume > 1.8x 24-period MA for institutional participation
# - HTF trend filter: 1w close > 1w EMA34 for long bias, < EMA34 for short bias
# - Entry: Long when close > H3 + volume filter + 1w uptrend; Short when close < L3 + volume filter + 1w downtrend
# - Exit: Close crosses back below H3 for long exit, above L3 for short exit
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# - Works in bull/bear: Camarilla levels act as support/resistance in ranging markets, volume confirms breakout validity, 1w trend ensures alignment with higher timeframe momentum

name = "12h_1d_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_ = prices['open'].values  # needed for Camarilla calculation
    
    # Pre-compute HTF data
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1w = df_1w['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels from prior 1d session
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume MA(24)
    volume_ma_24_1d = pd.Series(volume_1d).rolling(window=24, min_periods=24).mean().values
    volume_ma_24_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_24_1d)
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(30, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_ma_24_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 24-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.8 * volume_ma_24_1d_aligned[i]
        
        # Trend filter: 1w close > EMA34 for uptrend, < EMA34 for downtrend
        trend_up = close_1w[-1] > ema_34_1w[-1] if len(close_1w) > 0 else False
        trend_down = close_1w[-1] < ema_34_1w[-1] if len(close_1w) > 0 else False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: close > H3 + volume confirmation + 1w uptrend
            if (close[i] > camarilla_h3_aligned[i] and volume_confirm and trend_up):
                position = 1
                signals[i] = 0.25
            # Short entry: close < L3 + volume confirmation + 1w downtrend
            elif (close[i] < camarilla_l3_aligned[i] and volume_confirm and trend_down):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses back below H3 for long exit, above L3 for short exit
            if position == 1:  # Long position
                if close[i] < camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals