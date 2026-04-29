#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel from 1d data for breakout signals
# Long when price > upper band AND 1w EMA50 uptrend AND volume spike
# Short when price < lower band AND 1w EMA50 downtrend AND volume spike
# ATR-based stoploss to manage risk. Target: 30-100 trades over 4 years (7-25/year).

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian Channel (20) on 1d data
    period_donchian = 20
    highest_high = pd.Series(high).rolling(window=period_donchian, min_periods=period_donchian).max().values
    lowest_low = pd.Series(low).rolling(window=period_donchian, min_periods=period_donchian).min().values
    upper_band = highest_high
    lower_band = lowest_low
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Handle first value where roll creates NaN
    tr[0] = high[0] - low[0]
    atr[0] = tr[0]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(20, 50, 20, 14)  # warmup for Donchian, EMA50, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_band[i]
        curr_lower = lower_band[i]
        curr_ema50 = ema50_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        curr_atr = atr[i]
        
        # Trend regime: bullish if close > 1w EMA50, bearish if close < 1w EMA50
        is_bullish_regime = curr_close > curr_ema50
        is_bearish_regime = curr_close < curr_ema50
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price > upper band AND bullish regime
                if curr_close > curr_upper and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
                # Bearish entry: price < lower band AND bearish regime
                elif curr_close < curr_lower and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    atr_at_entry = curr_atr
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price < lower band OR stoploss hit OR regime changes to bearish
            stoploss_level = entry_price - 2.5 * atr_at_entry
            if (curr_close < curr_lower) or (curr_close < stoploss_level) or (not is_bullish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price > upper band OR stoploss hit OR regime changes to bullish
            stoploss_level = entry_price + 2.5 * atr_at_entry
            if (curr_close > curr_upper) or (curr_close > stoploss_level) or (not is_bearish_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals