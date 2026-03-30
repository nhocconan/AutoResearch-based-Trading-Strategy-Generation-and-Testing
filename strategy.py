#!/usr/bin/env python3
"""
Experiment #004: 1d Donchian Breakout + 1w EMA Trend + Volume Confirmation

HYPOTHESIS: Using 1d primary with 1w HTF EMA for maximum trend filtering.
Weekly EMA(21) provides strongest trend signal, eliminating countertrend trades.

CORE ELEMENTS (from DB winners):
1. Donchian(20) breakout on 1d - structural break detection
2. Weekly EMA(21) for trend direction - filters countertrend trades
3. Volume spike confirmation - institutional participation
4. ATR-based stop-loss - risk management

WHY IT SHOULD WORK IN BOTH MARKETS:
- 1d timeframe reduces noise vs 4h/12h, more significant breakouts
- 1w EMA catches major trend shifts, ignores intra-week noise
- 2022 crash: Weekly EMA bearish = no long entries during major下跌
- 2025 bear: Weekly EMA flat/bearish = selective short breakouts only
- Fewer trades but higher quality = better test generalization

TRADE COUNT ESTIMATE:
- ~250 1d bars/year
- Donchian breakout: ~5-8/year per direction
- 1w EMA filter: ~60% pass rate
- Volume spike: ~70% pass rate
- Final: ~15-25 trades/symbol/year → 60-100 total over 4 years ✓
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range - vectorized"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(values, period, min_periods=None):
    """Calculate EMA with proper min_periods"""
    if min_periods is None:
        min_periods = period
    return pd.Series(values).ewm(span=period, min_periods=min_periods, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: Weekly EMA for trend direction (call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = calculate_ema(df_1w['close'].values, period=21, min_periods=21)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel(20) for breakout structure
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # EMA for local trend (8-day)
    ema_8 = calculate_ema(close, period=8, min_periods=8)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 50  # Need enough for Donchian(20) + ATR alignment
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND DIRECTION (1w EMA) ===
        weekly_bullish = close[i] > ema_1w_aligned[i]
        weekly_bearish = close[i] < ema_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Use prior bar's channel to avoid look-ahead
        prev_upper = donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else np.nan
        prev_mid = donchian_mid[i-1] if i > 0 and not np.isnan(donchian_mid[i-1]) else np.nan
        
        # Bullish breakout: close above prior bar's upper channel + ATR buffer
        if not np.isnan(prev_upper):
            breakout_distance = close[i] - prev_upper
            bullish_breakout = breakout_distance > 0.5 * atr_14[i]  # Need 0.5 ATR confirmation
        else:
            bullish_breakout = False
        
        # Bearish breakout: close below prior bar's lower channel + ATR buffer
        if not np.isnan(prev_lower):
            breakout_distance = prev_lower - close[i]
            bearish_breakout = breakout_distance > 0.5 * atr_14[i]  # Need 0.5 ATR confirmation
        else:
            bearish_breakout = False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === LOCAL TREND (8 EMA alignment) ===
        local_bullish = close[i] > ema_8[i] if not np.isnan(ema_8[i]) else True
        local_bearish = close[i] < ema_8[i] if not np.isnan(ema_8[i]) else True
        
        # === MINIMUM HOLD: 2 bars (1d so 2 days minimum) ===
        min_hold = (i - entry_bar) >= 2
        
        # === EXITS ===
        if in_position:
            # Stop-loss: 2.5 ATR from entry
            if position_side > 0:
                stop_hit = low[i] < (entry_price - 2.5 * entry_atr)
                # Trailing stop: price below mid-channel
                trailing_exit = close[i] < prev_mid if not np.isnan(prev_mid) else False
            else:
                stop_hit = high[i] > (entry_price + 2.5 * entry_atr)
                # Trailing stop: price above mid-channel
                trailing_exit = close[i] > prev_mid if not np.isnan(prev_mid) else False
            
            # Exit on opposite breakout (trend reversal with volume)
            if position_side > 0 and bearish_breakout and vol_spike:
                reversal_exit = True
            elif position_side < 0 and bullish_breakout and vol_spike:
                reversal_exit = True
            else:
                reversal_exit = False
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            elif min_hold and (trailing_exit or reversal_exit):
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Weekly bullish + local bullish + bullish breakout + volume spike
            if weekly_bullish and local_bullish and bullish_breakout and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Weekly bearish + local bearish + bearish breakout + volume spike
            elif weekly_bearish and local_bearish and bearish_breakout and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals