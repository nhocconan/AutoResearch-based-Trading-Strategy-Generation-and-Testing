#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index + ADX regime filter + 12h trend confirmation
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when Bull Power > 0 AND Bear Power rising (improving) AND ADX > 25 (trending) AND 12h close > EMA50 (uptrend)
# - Short when Bear Power < 0 AND Bull Power falling (weakening) AND ADX > 25 (trending) AND 12h close < EMA50 (downtrend)
# - Exit when ADX < 20 (trend weakening) or power divergence fails
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Elder Ray measures buying/selling pressure behind price moves
# - ADX ensures we only trade in strong trends where Elder Ray works best
# - 12h EMA filter aligns with higher timeframe trend to avoid counter-trend whipsaws

name = "6h_12h_elder_ray_adx_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Primary 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # === ELDER RAY INDEX (6h) ===
    # EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # Power trends (rate of change)
    bull_power_change = np.diff(bull_power, prepend=bull_power[0])
    bear_power_change = np.diff(bear_power, prepend=bear_power[0])
    
    # === ADX (6h) for trend strength ===
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - np.roll(close, 1)[1:])
    tr3 = np.abs(low[1:] - np.roll(close, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    up_move = np.diff(high)
    down_move = -np.diff(low)  # negative of price drop
    up_move = np.concatenate([[0], up_move])
    down_move = np.concatenate([[0], down_move])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed ATR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(atr == 0, np.nan, atr)
    minus_di = 100 * minus_dm_smooth / np.where(atr == 0, np.nan, atr)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) == 0, 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 12h HTF Trend Filter ===
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align 12h EMA50 to 6h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13[i]) or np.isnan(adx[i]) or 
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Bull Power positive AND rising AND ADX > 25 AND 12h uptrend
            if (bull_power[i] > 0 and 
                bull_power_change[i] > 0 and  # Bull Power improving
                adx[i] > 25 and 
                close[i] > ema50_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Bear Power negative AND rising (more negative) AND ADX > 25 AND 12h downtrend
            elif (bear_power[i] > 0 and  # Bear Power positive means bearish pressure
                  bear_power_change[i] > 0 and  # Bear Power increasing (more bearish)
                  adx[i] > 25 and 
                  close[i] < ema50_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when trend weakens (ADX < 20) or power fails
            exit_long = (position == 1 and 
                        (adx[i] < 20 or  # Trend weakening
                         bull_power[i] <= 0 or  # Bull Power failed
                         bull_power_change[i] < 0))  # Bull Power deteriorating
            
            exit_short = (position == -1 and 
                         (adx[i] < 20 or  # Trend weakening
                          bear_power[i] <= 0 or  # Bear Power failed (no bearish pressure)
                          bear_power_change[i] < 0))  # Bear Power deteriorating
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals