#!/usr/bin/env python3
name = "6h_LiquiditySweep_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Identify recent swing highs/lows for liquidity sweep detection
    # Using 20-period lookback for swing points (appropriate for 6h timeframe)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, center=False).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, center=False).min().values
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, lookback, 4)  # Wait for EMA, swing points, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: liquidity sweep below recent low followed by reversal with volume in uptrend
            liq_sweep_low = low[i] < lowest_low[i-1]  # swept below recent low
            reversal = close[i] > open_prices[i] if 'open' in prices.columns else close[i] > low[i]  # bullish candle
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if liq_sweep_low and reversal and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: liquidity sweep above recent high followed by reversal with volume in downtrend
            elif high[i] > highest_high[i-1] and close[i] < open_prices[i] if 'open' in prices.columns else close[i] < high[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below swing low or volume drops
            if low[i] < lowest_low[i-1] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above swing high or volume drops
            if high[i] > highest_high[i-1] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Liquidity Sweep with 1d trend and volume confirmation
# - Liquidity sweeps (stop hunts) create high-probability reversal points
# - Long when price sweeps below recent low then reverses with volume in daily uptrend
# - Short when price sweeps above recent high then reverses with volume in daily downtrend
# - Volume confirmation (1.5x average) filters false breakouts
# - Works in both bull (buy sweeps in uptrend) and bear (sell sweeps in downtrend)
# - Exit when liquidity is taken in opposite direction or volume weakens
# - Position size 0.25 targets ~30-80 trades/year, avoiding fee drag
# - Novel approach: focuses on market microstructure (liquidity sweeps) rather than traditional breakouts
# - Uses 20-period swing points for appropriate 6h timeframe scale
# - Daily trend filter prevents trading against higher timeframe momentum
# - Designed for low frequency, high quality trades in ranging and trending markets