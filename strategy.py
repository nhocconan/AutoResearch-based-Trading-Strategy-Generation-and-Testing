#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss
# - Long: price breaks above 20-period Donchian high with volume > 1.5x 20-period avg volume
# - Short: price breaks below 20-period Donchian low with volume > 1.5x 20-period avg volume
# - Exit: ATR trailing stop (3x ATR) or opposite Donchian breakout
# - Uses 1d HTF trend filter: only trade in direction of 1d EMA(50) to avoid counter-trend whipsaws
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter

name = "4h_1d_donchian_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    # Load 1d data ONCE before loop for HTF trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 4h Donchian channels (20-period)
    # Donchian high = max(high, lookback=20)
    # Donchian low = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # first bar has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(50, n):  # Start after 50-bar warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Trend filter: only long if price > 1d EMA50, only short if price < 1d EMA50
        uptrend = close_price > ema_50_1d_aligned[i]
        downtrend = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long breakout: price breaks above Donchian high with volume and uptrend
        if close_price > donchian_high[i] and vol_confirm and uptrend:
            enter_long = True
        
        # Short breakdown: price breaks below Donchian low with volume and downtrend
        if close_price < donchian_low[i] and vol_confirm and downtrend:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: ATR trailing stop (3x ATR below highest high since entry)
            # or opposite Donchian breakdown
            if i > 0:
                atr_stop = max(atr_stop, high_price - 3.0 * atr[i])
                exit_long = close_price <= atr_stop or close_price < donchian_low[i]
            else:
                atr_stop = high_price - 3.0 * atr[i]
        elif position == -1:
            # Exit short: ATR trailing stop (3x ATR above lowest low since entry)
            # or opposite Donchian breakout
            if i > 0:
                atr_stop = min(atr_stop, low_price + 3.0 * atr[i])
                exit_short = close_price >= atr_stop or close_price > donchian_high[i]
            else:
                atr_stop = low_price + 3.0 * atr[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_price
            atr_stop = high_price - 3.0 * atr[i]  # initial stop
            signals[i] = 0.30
        elif enter_short and position != -1:
            position = -1
            entry_price = close_price
            atr_stop = low_price + 3.0 * atr[i]  # initial stop
            signals[i] = -0.30
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals