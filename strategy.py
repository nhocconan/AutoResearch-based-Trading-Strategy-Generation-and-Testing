#!/usr/bin/env python3
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
    
    # Load 1-week and 1-day data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly range for volatility filtering (more stable than daily)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_range = weekly_high - weekly_low
    
    # Calculate daily range for ATR-based volatility
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_range = daily_high - daily_low
    
    # Calculate 14-period ATR on daily data
    tr1 = np.abs(daily_high[1:] - daily_low[1:])
    tr2 = np.abs(daily_high[1:] - np.roll(daily_close, 1)[1:])
    tr3 = np.abs(daily_low[1:] - np.roll(daily_close, 1)[1:])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Pad first element
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly ATR percentile to filter for high volatility regimes
    atr_ratio = atr_14 / weekly_range  # Normalize by weekly range for stability
    atr_percentile = pd.Series(atr_ratio).rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Calculate daily ATR for entry sensitivity
    daily_atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Volume spike detection (using 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma20  # Slightly lower threshold for 12h timeframe
    
    # Calculate EMA50 on daily close for trend filter
    daily_close = df_1d['close'].values
    ema50_daily = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA200 on daily close for long-term trend
    ema200_daily = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 12-period RSI on daily close for momentum confirmation
    delta = np.diff(daily_close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/12, adjust=False, min_periods=12).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/12, adjust=False, min_periods=12).mean().values
    rs = avg_gain / avg_loss
    rsi_12 = 100 - (100 / (1 + rs))
    
    # Calculate weekly trend strength using price position relative to weekly range
    weekly_close = df_1w['close'].values
    weekly_position = (weekly_close - weekly_low) / weekly_range  # 0 = low, 1 = high
    
    # Align all HTF indicators to 12-hour timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_daily)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_daily)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_12)
    weekly_pos_aligned = align_htf_to_ltf(prices, df_1w, weekly_position)
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(weekly_pos_aligned[i]) or
            np.isnan(atr_percentile_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in high volatility regimes (top 30% of ATR ratio)
        high_vol_regime = atr_percentile_aligned[i] > 0.7
        
        if position == 0:
            # Long conditions: price above EMAs, bullish weekly position, RSI not overbought, volume spike
            if (close[i] > ema50_aligned[i] and 
                close[i] > ema200_aligned[i] and
                weekly_pos_aligned[i] > 0.6 and  # Weekly close in upper 40% of range
                rsi_aligned[i] < 70 and          # Not overbought
                vol_spike_aligned[i] and
                high_vol_regime):
                signals[i] = 0.25
                position = 1
            # Short conditions: price below EMAs, bearish weekly position, RSI not oversold, volume spike
            elif (close[i] < ema50_aligned[i] and 
                  close[i] < ema200_aligned[i] and
                  weekly_pos_aligned[i] < 0.4 and  # Weekly close in lower 40% of range
                  rsi_aligned[i] > 30 and          # Not oversold
                  vol_spike_aligned[i] and
                  high_vol_regime):
                signals[i] = -0.25
                position = -1
        else:
            # Dynamic exit based on ATR trailing stop
            if position == 1:
                # Trail stop at 2.5 * ATR below highest high since entry
                # Simplified: exit when price drops below EMA50 or RSI turns bearish
                if (close[i] < ema50_aligned[i] or 
                    rsi_aligned[i] < 40):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Trail stop at 2.5 * ATR above lowest low since entry
                # Simplified: exit when price rises above EMA50 or RSI turns bullish
                if (close[i] > ema50_aligned[i] or 
                    rsi_aligned[i] > 60):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_EMA_RSI_Vol_Spike_WeeklyTrend"
timeframe = "12h"
leverage = 1.0