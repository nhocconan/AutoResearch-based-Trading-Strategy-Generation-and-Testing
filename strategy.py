#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R reversal with 1w EMA trend filter and volume confirmation.
# Long when Williams %R crosses above -80 from below AND price > 1w EMA50 AND volume > 1.5x 20-period 1d average.
# Short when Williams %R crosses below -20 from above AND price < 1w EMA50 AND volume > 1.5x 20-period 1d average.
# Exit when Williams %R crosses the opposite threshold (-20 for long, -80 for short) or ATR-based stoploss (2*ATR).
# Uses discrete position size 0.25. Designed to catch reversals in ranging markets with trend filter.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while maintaining edge.
# Williams %R identifies overbought/oversold conditions; EMA50 filter ensures alignment with weekly trend.
# Volume confirmation reduces false signals. Works in both bull and bear markets by adapting to momentum extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Williams %R (14-period) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # === 1w Indicators: EMA50 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_1d)
    
    # === 1d Indicators: ATR for stoploss (14-period) ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(atr_1d[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        wr = williams_r[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1d[i]
        
        # Williams %R thresholds
        wr_oversold = -80.0
        wr_overbought = -20.0
        
        # Williams %R previous value for crossover detection
        wr_prev = williams_r[i-1] if i > 0 else wr
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above overbought threshold (-20) from below
            if wr > wr_overbought and wr_prev <= wr_overbought:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif price < entry_price - 2.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below oversold threshold (-80) from above
            if wr < wr_oversold and wr_prev >= wr_oversold:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif price > entry_price + 2.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R crosses above -80 from below AND price > weekly EMA50 AND volume spike
            wr_cross_up = wr > wr_oversold and wr_prev <= wr_oversold
            if wr_cross_up and price > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Williams %R crosses below -20 from above AND price < weekly EMA50 AND volume spike
            elif wr < wr_overbought and wr_prev >= wr_overbought and price < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_WilliamsR_1wEMA50_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0