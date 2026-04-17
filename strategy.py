#!/usr/bin/env python3
"""
1h_BreakerBlock_Sweep_Multitimeframe
Strategy: 1-hour Fair Value Gap (FVG) retest + breaker block (liquidity sweep) in direction of higher timeframe (4h/1d) trend.
Long: Bullish FVG formed (low[0] > high[-2]), price returns to fill 50% of gap, and liquidity sweep below prior low occurs, while 4h close > 1d EMA50.
Short: Bearish FVG formed (high[0] < low[-2]), price returns to fill 50% of gap, and liquidity sweep above prior high occurs, while 4h close < 1d EMA50.
Exit: Opposite FVG forms or price reaches 2x gap size in favor.
Position size: 0.20
Designed to work in both bull (continuation) and bear (mean reversion via liquidity sweeps) by aligning with HTF trend.
Timeframe: 1h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # FVG detection: 3-bar sequence where middle bar gaps away from first and third
    # Bullish FVG: low[i] > high[i-2] (gap up)
    # Bearish FVG: high[i] < low[i-2] (gap down)
    bullish_fvg = (low[2:] > high[:-2])  # aligned at index i (third bar)
    bearish_fvg = (high[2:] < low[:-2])
    
    # Pad to original length
    bullish_fvg = np.concatenate([np.full(2, False), bullish_fvg])
    bearish_fvg = np.concatenate([np.full(2, False), bearish_fvg])
    
    # FVG boundaries
    fvg_top = np.where(bullish_fvg, low, np.nan)      # for bullish: entry zone is from high[i-2] to low[i]
    fvg_bottom = np.where(bullish_fvg, high[:-2], np.nan)  # but we use high[i-2] as bottom, low[i] as top
    # Actually: for bullish FVG, gap is between high[i-2] and low[i]
    bullish_fvg_bottom = np.where(bullish_fvg, high[:-2], np.nan)
    bullish_fvg_top = np.where(bullish_fvg, low, np.nan)
    bearish_fvg_bottom = np.where(bearish_fvg, low, np.nan)
    bearish_fvg_top = np.where(bearish_fvg, high[:-2], np.nan)
    
    # 50% level of FVG for entry
    bullish_fvg_mid = (bullish_fvg_bottom + bullish_fvg_top) / 2
    bearish_fvg_mid = (bearish_fvg_bottom + bearish_fvg_top) / 2
    
    # Liquidity sweep: price takes out prior swing high/low then reverses
    # Bullish sweep: new low below prior low, then close > prior low (sweep and hold)
    # Bearish sweep: new high above prior high, then close < prior high
    lookback = 20
    roll_max = pd.Series(high).rolling(window=lookback, min_periods=1).max().values
    roll_min = pd.Series(low).rolling(window=lookback, min_periods=1).min().values
    
    # Sweep detection: price pierces level then closes back inside
    bullish_sweep = (low < roll_min) & (close > roll_min)  # took out low, closed back above
    bearish_sweep = (high > roll_max) & (close < roll_max)  # took out high, closed back below
    
    # HTF trend filter: 4h close vs 1d EMA50
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    close_series_4h = pd.Series(close_4h)
    ema50_4h = close_series_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_series_1d = pd.Series(close_1d)
    ema50_1d = close_series_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Trend: 4h close above/both 1d and 4h EMA50 for long, below for short
    trend_up = (close_4h > ema50_1d_aligned[::4][:len(close_4h)]) & (close_4h > ema50_4h)  # approximate alignment
    trend_down = (close_4h < ema50_1d_aligned[::4][:len(close_4h)]) & (close_4h < ema50_4h)
    # Simpler: use aligned 1d EMA on 1h chart directly
    trend_up = close > ema50_1d_aligned
    trend_down = close < ema50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # ensure EMA and FVG lookback ready
    
    for i in range(start_idx, n):
        # Track most recent FVG
        if bullish_fvg[i]:
            # Bullish FVG formed at i
            last_bullish_fvg_idx = i
            last_bullish_fvg_bottom = bullish_fvg_bottom[i]
            last_bullish_fvg_top = bullish_fvg_top[i]
            last_bullish_fvg_mid = bullish_fvg_mid[i]
        if bearish_fvg[i]:
            # Bearish FVG formed at i
            last_bearish_fvg_idx = i
            last_bearish_fvg_bottom = bearish_fvg_bottom[i]
            last_bearish_fvg_top = bearish_fvg_top[i]
            last_bearish_fvg_mid = bearish_fvg_mid[i]
        
        # Exit conditions: opposite FVG forms or price moves 2x gap in favor
        if position == 1:  # long
            # Exit if bearish FVG forms (new resistance above)
            if bearish_fvg[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price reaches 2x gap size above entry (take profit)
            elif 'last_bullish_fvg_idx' in locals() and i - last_bullish_fvg_idx < 50:  # valid FVG
                gap_size = last_bullish_fvg_top - last_bullish_fvg_bottom
                target = last_bullish_fvg_bottom + 2 * gap_size  # 2x risk-reward
                if close[i] >= target:
                    signals[i] = 0.0
                    position = 0
            # Otherwise hold
            else:
                signals[i] = 0.20
        elif position == -1:  # short
            # Exit if bullish FVG forms (new support below)
            if bullish_fvg[i]:
                signals[i] = 0.0
                position = 0
            # Exit if price reaches 2x gap size below entry
            elif 'last_bearish_fvg_idx' in locals() and i - last_bearish_fvg_idx < 50:
                gap_size = last_bearish_fvg_top - last_bearish_fvg_bottom
                target = last_bearish_fvg_top - 2 * gap_size
                if close[i] <= target:
                    signals[i] = 0.0
                    position = 0
            else:
                signals[i] = -0.20
        else:  # flat, look for entry
            # Long entry: bullish FVG retest + liquidity sweep + uptrend
            if ('last_bullish_fvg_idx' in locals() and 
                i - last_bullish_fvg_idx >= 1 and  # at least one bar after formation
                i - last_bullish_fvg_idx <= 20 and  # within reasonable retest window
                bullish_sweep[i] and  # liquidity sweep occurred
                low[i] <= last_bullish_fvg_mid and  # price retested to 50% level
                trend_up[i]):  # HTF trend up
                signals[i] = 0.20
                position = 1
                last_bullish_fvg_idx = -1  # invalidate after use
            # Short entry: bearish FVG retest + liquidity sweep + downtrend
            elif ('last_bearish_fvg_idx' in locals() and 
                  i - last_bearish_fvg_idx >= 1 and
                  i - last_bearish_fvg_idx <= 20 and
                  bearish_sweep[i] and
                  high[i] >= last_bearish_fvg_mid and
                  trend_down[i]):
                signals[i] = -0.20
                position = -1
                last_bearish_fvg_idx = -1
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_BreakerBlock_Sweep_Multitimeframe"
timeframe = "1h"
leverage = 1.0