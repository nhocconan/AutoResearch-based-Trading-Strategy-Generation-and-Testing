#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above 20-day high with 1w EMA34 uptrend and volume > 1.5x 20-day average.
# Short when price breaks below 20-day low with 1w EMA34 downtrend and volume confirmation.
# Uses discrete sizing 0.25. ATR-based stoploss (signal→0 when price moves against position by 2.5*ATR).
# Primary timeframe: 1d, HTF: 1w for EMA trend filter.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag and improve generalization.
# No session filter - daily timeframe captures full market action.

name = "1d_Donchian20_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) from daily OHLC
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels (already 1d, no need to align)
    donchian_high = high_roll
    donchian_low = low_roll
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = 34  # warmup for EMA34 and ATR
    
    for i in range(start_idx, n):
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        if vol_ma <= 0:
            volume_confirm = False
        else:
            volume_confirm = volume[i] > (vol_ma * 1.5)
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian high AND bullish 1w trend AND volume confirmation
            if (curr_high > donchian_high[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Breakout below Donchian low AND bearish 1w trend AND volume confirmation
            elif (curr_low < donchian_low[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low OR 1w trend turns bearish
            elif (curr_low < donchian_low[i] or 
                  close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high OR 1w trend turns bullish
            elif (curr_high > donchian_high[i] or 
                  close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals