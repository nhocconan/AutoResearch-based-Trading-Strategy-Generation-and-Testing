#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume confirmation.
# Long when Williams %R(14) crosses above -80 (oversold bounce) AND 1d close > 1d EMA50 (uptrend) AND 6h volume > 1.5x 20-period average.
# Short when Williams %R(14) crosses below -20 (overbought rejection) AND 1d close < 1d EMA50 (downtrend) AND 6h volume > 1.5x 20-period average.
# Exit on opposite Williams %R cross (-50 for mean reversion) or ATR stop (2*ATR).
# Uses discrete position size 0.25. Works in ranging markets via mean reversion and in trends via pullback entries.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Williams %R (14-period) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # === 1d Indicators: EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close_1d > ema_50_1d  # HTF trend direction
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    
    # === 6h Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    # === 6h ATR for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50)
    warmup = 60
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_6h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        atr_val = atr_6h_raw[i]
        uptrend = uptrend_1d_aligned[i] > 0.5  # boolean from aligned array
        
        # Williams %R cross signals (using previous bar to avoid look-ahead)
        wr_prev = williams_r[i-1]
        cross_above_80 = wr_prev <= -80 and wr > -80   # oversold bounce
        cross_below_20 = wr_prev >= -20 and wr < -20   # overbought rejection
        cross_above_50 = wr_prev <= -50 and wr > -50   # mean reversion exit long
        cross_below_50 = wr_prev >= -50 and wr < -50   # mean reversion exit short
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if mean reversion signal (cross above -50) or ATR stop
            if cross_above_50:
                exit_signal = True
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if mean reversion signal (cross below -50) or ATR stop
            if cross_below_50:
                exit_signal = True
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R cross above -80 (oversold bounce) AND uptrend AND volume spike
            if cross_above_80 and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R cross below -20 (overbought rejection) AND downtrend AND volume spike
            elif cross_below_20 and not uptrend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            # Hold current position
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR_1dTrend_VolumeSpike_ATRStop_V1"
timeframe = "6h"
leverage = 1.0