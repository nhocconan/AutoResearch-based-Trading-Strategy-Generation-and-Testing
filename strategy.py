# Hyperscale: 6h 1D/1W Confluence Breakout Strategy
# Hypothesis: 6-hour breakouts aligned with daily trend (EMA34) and weekly momentum (RSI>50) capture sustained moves
# while avoiding counter-trend noise. Weekly RSI filter ensures we only trade with higher timeframe momentum.
# Volume confirmation filters low-quality breakouts. Designed for 12-30 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and weekly for momentum filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily EMA34 trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Weekly RSI momentum filter (14-period)
    close_1w = pd.Series(df_1w['close'].values)
    delta = close_1w.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = rsi_1w.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w, additional_delay_bars=0)
    
    # 6-hour Donchian breakout channels (20-period)
    high_6h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_6h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(high_6h[i]) or np.isnan(low_6h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_1d_aligned[i]
        rsi_momentum = rsi_1w_aligned[i]
        upper_channel = high_6h[i]
        lower_channel = low_6h[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above upper channel with daily uptrend and weekly momentum
            if price > upper_channel and vol_ok and price > ema_trend and rsi_momentum > 50:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with daily downtrend and weekly weakness
            elif price < lower_channel and vol_ok and price < ema_trend and rsi_momentum < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price returns below daily EMA or weekly momentum fades
            if price < ema_trend or rsi_momentum < 45:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns above daily EMA or weekly momentum recovers
            if price > ema_trend or rsi_momentum > 55:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1D_1W_Confluence_Breakout"
timeframe = "6h"
leverage = 1.0