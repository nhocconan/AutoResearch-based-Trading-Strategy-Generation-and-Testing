#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter and volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) on 6h
# - Entry Long: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1d close > 1d EMA(50) (bullish regime) AND 6h volume > 1.5x 20-period average
# - Entry Short: Bear Power < 0 AND Bull Power > 0 (bearish momentum) AND 1d close < 1d EMA(50) (bearish regime) AND 6h volume > 1.5x 20-period average
# - Exit: Close-based reversal - exit long when Bear Power >= 0, exit short when Bull Power <= 0
# - Stoploss: ATR-based - exit when price moves against position by 2.5 * ATR(14) on 6h
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year)
# - Elder Ray measures bull/bear power relative to EMA, 1d EMA(50) filters regime, volume confirms participation
# - Works in bull markets via long signals when bull power dominates, in bear markets via short signals when bear power dominates

name = "6h_1d_elderray_volume_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 1d data for regime filter
    close_1d = df_1d['close'].values
    
    # Calculate 6h EMA(13) for Elder Ray
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_6h - ema13_6h  # High - EMA13
    bear_power = low_6h - ema13_6h   # Low - EMA13
    
    # Calculate 6h volume moving average (20-period)
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA(50) for regime filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h ATR (14-period) for stoploss
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr1_6h[0] = 0
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, prices, bull_power)  # 6h data already aligned
    bear_power_aligned = align_htf_to_ltf(prices, prices, bear_power)  # 6h data already aligned
    volume_ma_aligned = align_htf_to_ltf(prices, prices, volume_ma_20_6h)  # 6h data already aligned
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    atr_6h_aligned = align_htf_to_ltf(prices, prices, atr_6h)  # 6h data already aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h close
        close_price = close_6h[i]
        
        # Get current 6h volume for confirmation
        volume_6h_current = volume_6h[i]
        volume_confirmation = volume_6h_current > 1.5 * volume_ma_aligned[i]
        
        # Regime filter: 1d close vs 1d EMA50
        close_1d_current = close_1d[i // 24] if i // 24 < len(close_1d) else close_1d[-1]  # Approximate 1d close for current 6h bar
        ema50_1d_current = ema50_1d_aligned[i]
        bullish_regime = close_1d_current > ema50_1d_current
        bearish_regime = close_1d_current < ema50_1d_current
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND bullish regime AND volume confirmation
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] < 0 and 
                bullish_regime and 
                volume_confirmation):
                position = 1
                entry_price = close_price
                signals[i] = 0.25
            # Short entry: Bear Power < 0 AND Bull Power > 0 (bearish momentum) AND bearish regime AND volume confirmation
            elif (bear_power_aligned[i] < 0 and 
                  bull_power_aligned[i] > 0 and 
                  bearish_regime and 
                  volume_confirmation):
                position = -1
                entry_price = close_price
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or stoploss
            # Calculate stoploss level
            if position == 1:  # Long position
                stop_loss = entry_price - 2.5 * atr_6h_aligned[i]
                # Exit conditions: Bear Power >= 0 (loss of bullish momentum) OR stoploss hit
                if bear_power_aligned[i] >= 0 or close_price <= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                stop_loss = entry_price + 2.5 * atr_6h_aligned[i]
                # Exit conditions: Bull Power <= 0 (loss of bearish momentum) OR stoploss hit
                if bull_power_aligned[i] <= 0 or close_price >= stop_loss:
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals