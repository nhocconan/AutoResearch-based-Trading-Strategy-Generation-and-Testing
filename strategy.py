#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h ADX trend strength filter with 4h EMA direction and volume confirmation
# - Primary: 1h ADX(14) > 25 indicates strong trend (works in both bull/bear markets)
# - HTF trend: 4h EMA(21) slope determines trend direction (rising = long bias, falling = short bias)
# - HTF volume: 1d volume > 1.5x 20-period MA for institutional participation
# - Session filter: 08-20 UTC to avoid low-liquidity hours
# - Long: ADX > 25 + rising 4h EMA slope + volume spike + session
# - Short: ADX > 25 + falling 4h EMA slope + volume spike + session
# - Exit: ADX < 20 (trend weakening) or opposite EMA crossover
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# - Works in bull/bear: ADX filters weak/choppy markets, volume confirms participation, EMA slope gives direction

name = "1h_4h_1d_adx_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Pre-compute HTF data
    close_4h = df_4h['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1h ADX(14)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth the values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h EMA(21) and its slope
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate EMA slope (rate of change over 3 periods)
    ema_slope = np.zeros_like(ema_4h_aligned)
    for i in range(3, len(ema_4h_aligned)):
        if not np.isnan(ema_4h_aligned[i]) and not np.isnan(ema_4h_aligned[i-3]):
            ema_slope[i] = (ema_4h_aligned[i] - ema_4h_aligned[i-3]) / 3
        else:
            ema_slope[i] = np.nan
    
    # Calculate 1d volume MA(20)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(adx[i]) or np.isnan(ema_4h_aligned[i]) or np.isnan(ema_slope[i]) or
            np.isnan(volume_ma_20_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: ADX > 25 + rising 4h EMA slope + volume spike + session
            if (adx[i] > 25 and ema_slope[i] > 0 and volume_confirm):
                position = 1
                signals[i] = 0.20
            # Short entry: ADX > 25 + falling 4h EMA slope + volume spike + session
            elif (adx[i] > 25 and ema_slope[i] < 0 and volume_confirm):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: ADX < 20 (trend weakening) or EMA slope changes direction
            if position == 1:  # Long position
                if adx[i] < 20 or ema_slope[i] < 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if adx[i] < 20 or ema_slope[i] > 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals