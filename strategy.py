#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, price > 1d EMA34, volume > 1.5x 20-bar average
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, price < 1d EMA34, volume > 1.5x 20-bar average
# Uses 6h timeframe targeting 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
# Elder Ray measures trend strength via price-EMA divergence; EMA34 filter ensures higher-timeframe alignment; volume avoids chop.

name = "6h_ElderRay_BullBearPower_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components (6h)
    # EMA(13) for power calculations
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    # Bull Power = High - EMA(13)
    bull_power = high - ema_13
    # Bear Power = Low - EMA(13)
    bear_power = low - ema_13
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20, 13)  # EMA34, volume MA20, EMA13
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema34_val = ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 and rising, Bear Power < 0, price above 1d EMA34, volume spike
            if i > start_idx:
                bull_rising = bull_val > bull_power[i-1]
                bear_falling = bear_val < bear_power[i-1]
                if bull_val > 0 and bear_val < 0 and bull_rising and bear_falling and price > ema34_val and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                # Short entry: Bear Power < 0 and falling, Bull Power > 0, price below 1d EMA34, volume spike
                elif bear_val < 0 and bull_val > 0 and not bull_rising and not bear_falling and price < ema34_val and vol_confirm:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or Elder Ray weakening
            # ATR-based stoploss: 2.0 * ATR below entry (using 6h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or when Elder Ray shows weakening (Bull Power falling or Bear Power rising)
            if i > start_idx:
                bull_falling = bull_val < bull_power[i-1]
                bear_rising = bear_val > bear_power[i-1]
                if price < stop_loss or bull_falling or bear_rising:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or Elder Ray weakening
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or when Elder Ray shows weakening (Bear Power rising or Bull Power falling)
            if i > start_idx:
                bull_falling = bull_val < bull_power[i-1]
                bear_rising = bear_val > bear_power[i-1]
                if price > stop_loss or not bull_falling or not bear_rising:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals