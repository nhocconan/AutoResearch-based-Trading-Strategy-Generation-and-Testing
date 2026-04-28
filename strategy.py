#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels
# work well in ranging markets. Combined with 12h EMA34 trend filter to avoid counter-trend
# trades and volume confirmation to ensure genuine momentum. Designed for choppy/range-bound
# markets (2025 BTC/ETH bear/range conditions) while still capturing trends.
# Target: 12-35 trades/year via Williams %R extremes (below -80 for long, above -20 for short)
# with trend and volume filters to reduce false signals.

name = "6h_WilliamsR_Reversal_12hEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R parameters
    williams_period = 14
    
    # Calculate Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using rolling window for highest high and lowest low
    highest_high = pd.Series(high).rolling(window=williams_period, min_periods=williams_period).max().values
    lowest_low = pd.Series(low).rolling(window=williams_period, min_periods=williams_period).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    williams_r = np.where(hl_range != 0, ((highest_high - close) / hl_range) * -100, -50)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 12h close for trend filter
    close_12h = pd.Series(df_12h['close'])
    ema34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA34 to 6h timeframe (completed 12h candles only)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: >1.8x 30-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_30 = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > 1.8 * volume_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(30, 34)  # Need sufficient history for volume MA and EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(volume_ma_30[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        wr = williams_r[i]
        ema34_val = ema34_12h_aligned[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long reversal: Williams %R crosses above -80 from oversold AND 12h EMA34 uptrend AND volume spike
            if wr > -80 and wr < -20 and price > ema34_val and vol_confirm:
                # Additional confirmation: previous bar was below -80
                if i > 0 and williams_r[i-1] <= -80:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            # Short reversal: Williams %R crosses below -20 from overbought AND 12h EMA34 downtrend AND volume spike
            elif wr < -20 and wr > -80 and price < ema34_val and vol_confirm:
                # Additional confirmation: previous bar was above -20
                if i > 0 and williams_r[i-1] >= -20:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on Williams %R > -20 (overbought) or stoploss
            # ATR-based stoploss: 2.0 * ATR below entry (using 6h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on Williams %R > -20 (overbought) or stoploss
            if wr > -20 or price < stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on Williams %R < -80 (oversold) or stoploss
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on Williams %R < -80 (oversold) or stoploss
            if wr < -80 or price > stop_loss:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals