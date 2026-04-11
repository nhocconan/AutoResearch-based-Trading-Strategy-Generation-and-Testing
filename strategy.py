#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout + 1d/1w trend filter + volume confirmation
# - Long when price breaks above 12h Donchian upper channel (20-period) + price > 1d EMA50 + price > 1w EMA200 + volume > 2x 20-period average
# - Short when price breaks below 12h Donchian lower channel + price < 1d EMA50 + price < 1w EMA200 + volume > 2x 20-period average
# - Exit when price returns to Donchian middle (mean) or opposite breakout occurs
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) for 12h timeframe
# - Works in bull markets via trend-following breakouts and in bear via short breakdowns with volume confirmation
# - Multi-timeframe trend filters (1d EMA50, 1w EMA200) reduce false signals in choppy markets

name = "12h_1d_1w_donchian_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d and 1w data ONCE before loop for trend filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 200:
        return signals
    
    # Pre-compute 1d EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute 1w EMA200
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # Pre-compute 12h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Breakout conditions
        breakout_up = price_high > donchian_high[i]  # Price breaks above upper channel
        breakout_down = price_low < donchian_low[i]   # Price breaks below lower channel
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Trend filters: price relative to EMA50 (1d) and EMA200 (1w)
        price_above_ema50 = price_close > ema50_1d_aligned[i]
        price_below_ema50 = price_close < ema50_1d_aligned[i]
        price_above_ema200 = price_close > ema200_1w_aligned[i]
        price_below_ema200 = price_close < ema200_1w_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bullish breakout + above both EMAs + volume confirmation
        if breakout_up and price_above_ema50 and price_above_ema200 and vol_confirm:
            enter_long = True
        
        # Short: Bearish breakout + below both EMAs + volume confirmation
        if breakout_down and price_below_ema50 and price_below_ema200 and vol_confirm:
            enter_short = True
        
        # Exit conditions: return to middle or opposite breakout
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price returns to middle OR bearish breakout occurs
            exit_long = (price_close <= donchian_middle[i]) or breakout_down
        elif position == -1:
            # Exit short if price returns to middle OR bullish breakout occurs
            exit_short = (price_close >= donchian_middle[i]) or breakout_up
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals