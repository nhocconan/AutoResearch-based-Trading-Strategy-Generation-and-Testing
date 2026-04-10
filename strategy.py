#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation
# - Primary: 6h Elder Ray (EMA13-based Bull Power = Close - EMA13, Bear Power = EMA13 - Close)
# - HTF trend filter: 12h close > 12h EMA34 for uptrend bias, < EMA34 for downtrend bias
# - Volume confirmation: 6h volume > 1.5x 20-period MA for institutional participation
# - Entry: Long when Bull Power > 0 AND volume confirmation AND 12h uptrend; Short when Bear Power > 0 AND volume confirmation AND 12h downtrend
# - Exit: Opposite signal triggers (Bear Power > 0 for long exit, Bull Power > 0 for short exit)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Elder Ray measures power behind moves, volume confirms validity, 12h trend ensures alignment with higher timeframe momentum
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_12h_elder_ray_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate Elder Ray components
    bull_power = close - ema13  # Positive = bulls in control
    bear_power = ema13 - close  # Positive = bears in control
    
    # Calculate 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate 6h volume MA(20)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(40, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema34_12h_aligned[i]) or
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period MA
        volume_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        # Trend filter: 12h close > EMA34 for uptrend, < EMA34 for downtrend
        trend_up = close_12h[-1] > ema34_12h[-1] if len(close_12h) > 0 else False
        trend_down = close_12h[-1] < ema34_12h[-1] if len(close_12h) > 0 else False
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 + volume confirmation + 12h uptrend
            if (bull_power[i] > 0 and volume_confirm and trend_up):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power > 0 + volume confirmation + 12h downtrend
            elif (bear_power[i] > 0 and volume_confirm and trend_down):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Opposite power becomes positive (market reverses)
            if position == 1:  # Long position
                if bear_power[i] > 0:  # Bears taking control
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if bull_power[i] > 0:  # Bulls taking control
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals