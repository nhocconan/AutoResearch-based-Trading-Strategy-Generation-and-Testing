#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA50 trend filter.
# Long when price breaks above upper Donchian channel with volume spike and 12h EMA50 uptrend.
# Short when price breaks below lower Donchian channel with volume spike and 12h EMA50 downtrend.
# Uses tight entry conditions to limit trades (target: 20-50/year) and ATR-based stoploss.
# Works in both bull and bear markets by aligning with higher timeframe trend.

name = "4h_Donchian20_12hTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        upper_channel[i] = np.max(high[i - lookback + 1:i + 1])
        lower_channel[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate ATR(14) for stoploss and volume MA
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume regime: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        ema_trend = ema_50_12h_aligned[i]
        
        # Regime: bull if close > 12h EMA50, bear if close < 12h EMA50
        is_bull_regime = close_val > ema_trend
        is_bear_regime = close_val < ema_trend
        
        # Breakout conditions
        breakout_up = close_val > upper_channel[i]
        breakout_down = close_val < lower_channel[i]
        
        # Generate signals
        if position == 0:
            # Long entry: bullish breakout in bull regime with volume spike
            if is_bull_regime and breakout_up and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short entry: bearish breakout in bear regime with volume spike
            elif is_bear_regime and breakout_down and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long position: trail stop or exit on breakdown
            signals[i] = 0.25
            # Stoploss: 2 * ATR below entry price
            if close_val < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit on bearish breakout or regime change to bear
            elif breakout_down or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short position: trail stop or exit on breakup
            signals[i] = -0.25
            # Stoploss: 2 * ATR above entry price
            if close_val > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit on bullish breakout or regime change to bull
            elif breakout_up or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals