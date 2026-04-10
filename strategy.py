#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h/1d regime filter
# - Primary: 1h price breaks Camarilla H3/L3 levels from prior 4h bar (intraday momentum)
# - HTF regime: 4h close > 1d EMA(50) for bull bias, < for bear bias (multi-TF alignment)
# - Volume filter: 1h volume > 1.5x 20-period MA to confirm participation
# - Session filter: 08-20 UTC to focus on liquid London/NY overlap
# - Long: price > H3 + bull regime + volume spike + session
# - Short: price < L3 + bear regime + volume spike + session
# - Exit: price returns to Camarilla Pivot level (mean reversion) or regime flips
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# - Works in bull/bear: Camarilla levels adapt to volatility, regime filter avoids counter-trend, volume confirms

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    close_1d = df_1d['close'].values
    
    # Calculate 1h volume MA(20)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA(50) for regime filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(50, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(volume_ma_20[i]) or np.isnan(ema_50_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Get prior completed 4h bar for Camarilla calculation
        idx_4h = i // 16  # 16x 1h bars per 4h
        if idx_4h < 1:
            signals[i] = 0.0
            continue
            
        # Prior completed 4h bar OHLC (use index -1 for completed bar)
        h_4h = df_4h['high'].iloc[idx_4h - 1]
        l_4h = df_4h['low'].iloc[idx_4h - 1]
        c_4h = df_4h['close'].iloc[idx_4h - 1]
        
        # Calculate Camarilla levels for prior 4h bar
        range_4h = h_4h - l_4h
        if range_4h <= 0:
            signals[i] = 0.0
            continue
            
        camarilla_pivot = (h_4h + l_4h + c_4h) / 3
        camarilla_h3 = camarilla_pivot + (range_4h * 1.1 / 4)
        camarilla_l3 = camarilla_pivot - (range_4h * 1.1 / 4)
        
        # Volume confirmation: current 1h volume > 1.5x 20-period MA
        volume_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        # Regime filter: 4h close vs 1d EMA(50)
        bull_regime = close_4h[idx_4h - 1] > ema_50_1d_aligned[i]  # use aligned 1d EMA at current 1h
        bear_regime = close_4h[idx_4h - 1] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > H3 + bull regime + volume spike + session
            if (close[i] > camarilla_h3 and bull_regime and volume_confirm):
                position = 1
                signals[i] = 0.20
            # Short entry: price < L3 + bear regime + volume spike + session
            elif (close[i] < camarilla_l3 and bear_regime and volume_confirm):
                position = -1
                signals[i] = -0.20
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price returns to Camarilla Pivot or regime flips
            if position == 1:  # Long position
                if close[i] < camarilla_pivot or not bull_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # position == -1 (Short position)
                if close[i] > camarilla_pivot or not bear_regime:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
    
    return signals