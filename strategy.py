# NOTE: This strategy was developed but did not pass evaluation in the latest round.
# It is kept in the history for reference and learning.
# Final metrics: train_sharpe=0.583, test_sharpe=-0.141
# Strategy: 4h_PriceAction_Momentum_Consolidation_Breakout
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour price action momentum with consolidation breakout detection
# Strategy identifies periods of low volatility (consolidation) followed by
# momentum breakouts in the direction of the 12-hour trend.
# Uses Bollinger Band width for consolidation detection and RSI for momentum.
# In strong trends (ADX > 25), trades breakouts with volume confirmation.
# Designed to work in both bull and bear markets by following the trend.
# Target: 20-35 trades/year to minimize fee decay while capturing strong moves.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Trend Indicators ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA34 for trend direction
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 12h ADX for trend strength
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    up_move = high_12h - np.concatenate([[high_12h[0]], high_12h[:-1]])
    down_move = np.concatenate([[low_12h[0]], low_12h[:-1]]) - low_12h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * plus_dm_smooth / (atr_12h + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_12h + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 4h Consolidation and Momentum ===
    # Bollinger Bands for volatility squeeze detection
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # RSI for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(bb_width[i]) or np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Consolidation: Bollinger Band width in lowest 20% of last 50 periods
        bb_width_low = bb_width[i] < np.percentile(bb_width[max(0, i-50):i+1], 20)
        
        # Momentum: RSI > 55 for bullish, < 45 for bearish
        mom_bullish = rsi[i] > 55
        mom_bearish = rsi[i] < 45
        
        # Trend filter: 12h trend direction and strength
        uptrend = close_12h[-1] > ema34_12h[-1] if len(close_12h) > 0 else False  # Use last known 12h value
        strong_trend = adx_12h_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.3x average
        vol_confirm = volume[i] > vol_ma_20[i] * 1.3
        
        # Breakout: price outside Bollinger Bands
        breakout_up = close[i] > upper_bb[i]
        breakout_down = close[i] < lower_bb[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Need consolidation breakout with momentum and trend alignment
            if bb_width_low and vol_confirm and strong_trend:
                if mom_bullish and breakout_up and uptrend:
                    signals[i] = 0.25
                    position = 1
                    continue
                elif mom_bearish and breakout_down and not uptrend:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic: exit when momentum reverses or volatility expands
        elif position == 1:
            # Exit long if bearish momentum or volatility expansion
            if mom_bearish or bb_width[i] > np.percentile(bb_width[max(0, i-20):i+1], 80):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if bullish momentum or volatility expansion
            if mom_bullish or bb_width[i] > np.percentile(bb_width[max(0, i-20):i+1], 80):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_PriceAction_Momentum_Consolidation_Breakout"
timeframe = "4h"
leverage = 1.0