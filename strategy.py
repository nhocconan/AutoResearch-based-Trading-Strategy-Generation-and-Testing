# 4H_ADX_DMI_TREND_FOLLOW: 4h trend following with ADX trend strength filter and DMI crossover
# Uses ADX > 25 to filter strong trends and DMI crossover for entry/exit
# Works in both bull and bear markets by following established trends
# Target: 20-50 trades/year to minimize fee drag
import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4h ADX and DMI (14) ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilders_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # === 4h EMA (21) for trend bias ===
    close_series = pd.Series(close)
    ema_21 = close_series.ewm(span=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(ema_21[i])):
            signals[i] = 0.0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Strong trend required: ADX > 25
            if adx[i] > 25:
                # Long: +DI crosses above -DI and price above EMA21
                if plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1] and close[i] > ema_21[i]:
                    signals[i] = 0.25
                    position = 1
                    continue
                # Short: -DI crosses above +DI and price below EMA21
                elif minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1] and close[i] < ema_21[i]:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long: -DI crosses above +DI or ADX weakens (< 20) or price crosses below EMA21
            if (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]) or adx[i] < 20 or close[i] < ema_21[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: +DI crosses above -DI or ADX weakens (< 20) or price crosses above EMA21
            if (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]) or adx[i] < 20 or close[i] > ema_21[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4H_ADX_DMI_TREND_FOLLOW"
timeframe = "4h"
leverage = 1.0