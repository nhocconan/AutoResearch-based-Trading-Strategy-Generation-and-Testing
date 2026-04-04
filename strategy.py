#!/usr/bin/env python3
"""
Experiment #5274: 1h Donchian Breakout with 4h/1d Trend Filter and Volume Spike
HYPOTHESIS: On 1h timeframe, breakouts above/below 20-period Donchian channels 
with volume confirmation (1.5x average volume) filtered by 4h EMA20 trend 
and 1d EMA50 regime will capture strong momentum moves while minimizing whipsaws. 
The 4h/1d filters ensure we only trade in the direction of the higher timeframe trend, 
reducing false breakouts. Session filter (08-20 UTC) avoids low liquidity periods. 
Position size fixed at 0.20 to control drawdown. Target: 60-150 total trades over 4 years 
(15-37/year) to balance opportunity with fee drag (0.10% round trip per trade).
Works in bull markets by buying breakouts in uptrends and in bear markets by selling 
breakouts in downtrends, while avoiding ranging conditions where Donchian channels 
are narrow and breakouts fail.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5274_1h_donchian_breakout_4h_1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for EMA20 trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 20:
        ema_20 = pd.Series(df_4h['close']).ewm(span=20, min_periods=20, adjust=False).mean().shift(1).values
        ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20)
    else:
        ema_20_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for EMA50 regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().shift(1).values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    # Channel width for volatility filter
    donchian_width = donchian_high - donchian_low
    
    # === 1h Indicators: Volume Spike (1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 20, 50)  # Donchian, volume MA, EMA50 warmup
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position when price retraces to midpoint of Donchian channel ---
        if in_position:
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            
            if position_side > 0:  # Long position
                if price < donchian_mid:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                if price > donchian_mid:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Breakout conditions
        breakout_up = price > donchian_high[i]
        breakout_down = price < donchian_low[i]
        
        # Trend and regime filters from HTF
        trend_bullish = price > ema_20_4h_aligned[i]
        trend_bearish = price < ema_20_4h_aligned[i]
        regime_bullish = price > ema_50_1d_aligned[i]
        regime_bearish = price < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry conditions: Breakout + HTF alignment + volume
        if breakout_up and trend_bullish and regime_bullish and vol_confirm:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif breakout_down and trend_bearish and regime_bearish and vol_confirm:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals