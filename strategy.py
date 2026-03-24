#!/usr/bin/env python3
"""
Experiment #1517: 1d Primary + 1w HTF — KAMA Adaptive Trend + RSI Timing + Volume Confirm

Hypothesis: Based on #1516 success (12h KAMA+CRSI+ADX Sharpe=0.133), scaling to 1d with 
simpler RSI (not CRSI) should generate MORE trades while maintaining quality. KAMA adapts 
to volatility better than HMA/EMA, reducing whipsaws in choppy crypto markets.

Key insights from 1100+ failed strategies:
1. CRSI+CHOP complex regimes = negative Sharpe (#1511, #1512, #1514 all failed)
2. KAMA works better than HMA for crypto (#1516 kept with +47.3% return)
3. SIMPLER entry conditions = more trades (critical for meeting min trade requirements)
4. Volume confirmation adds edge without over-filtering
5. 1d timeframe needs LOOSE RSI bands (35-65) to ensure 30+ trades/train

Design:
- 1w KAMA(21) for macro trend direction (HTF filter - very loose)
- 1d KAMA(21) for primary trend + entry trigger (KAMA slope change)
- 1d RSI(14) for timing (loose: 35-65 range ensures trades happen)
- Volume ratio (taker_buy/volume) for confirmation (>0.45 long, <0.55 short)
- ATR(14) 2.0x trailing stop (tighter than 2.5x for better risk/reward)
- Position size 0.30 (discrete: 0.0, ±0.30)
- Target: 40-80 trades/train (4 years), 10-20 trades/test (15 months)

Timeframe: 1d (as required by experiment)
HTF: 1w (weekly trend bias)
Position Size: 0.30 (discrete levels to minimize fee churn)
Target: Sharpe > 0.618 (beat current best), DD < -30%, trades >= 30 train / >= 3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_rsi_volume_1w_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=21, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts to market noise - smooth in choppy markets, responsive in trends
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[max(0, i - period):i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if np.isnan(kama[i - 1]):
            continue
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(taker_buy_volume, volume):
    """Taker buy volume ratio (0-1, >0.5 = bullish pressure)"""
    ratio = np.zeros(len(volume))
    mask = volume > 1e-10
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w KAMA for trend bias
    kama_1w_raw = calculate_kama(df_1w['close'].values, period=21)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w_raw)
    
    # Calculate primary (1d) indicators
    kama_1d = calculate_kama(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    # KAMA slope (direction change detection)
    kama_slope = np.zeros(n)
    for i in range(2, n):
        if not np.isnan(kama_1d[i]) and not np.isnan(kama_1d[i-1]):
            kama_slope[i] = kama_1d[i] - kama_1d[i-1]
        else:
            kama_slope[i] = np.nan
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30  # Appropriate size for 1d (40-80 trades/year target)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(kama_1d[i]) or np.isnan(kama_slope[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_1w_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w KAMA) - primary direction bias ===
        weekly_bull = close[i] > kama_1w_aligned[i]
        weekly_bear = close[i] < kama_1w_aligned[i]
        
        # === PRIMARY TREND (1d KAMA) - confirmation ===
        daily_bull = close[i] > kama_1d[i]
        daily_bear = close[i] < kama_1d[i]
        
        # === KAMA SLOPE - momentum direction ===
        kama_bull = kama_slope[i] > 0
        kama_bear = kama_slope[i] < 0
        
        # === RSI TIMING - LOOSE bands for MORE trades ===
        # Long: RSI not overbought (allows entries throughout uptrend)
        rsi_ok_long = rsi[i] < 65.0
        # Short: RSI not oversold (allows entries throughout downtrend)
        rsi_ok_short = rsi[i] > 35.0
        
        # === VOLUME CONFIRMATION ===
        vol_bull = vol_ratio[i] > 0.45  # slight buy pressure
        vol_bear = vol_ratio[i] < 0.55  # slight sell pressure
        
        # === DESIRED SIGNAL - SIMPLIFIED FOR 1d (ensures trades) ===
        desired_signal = 0.0
        
        # LONG: Multiple confluence paths (any one triggers)
        # Path 1: Strong trend (1w + 1d + KAMA slope all bull) + RSI ok
        if weekly_bull and daily_bull and kama_bull and rsi_ok_long:
            desired_signal = BASE_SIZE
        # Path 2: 1w bull + 1d bull + volume confirm (looser)
        elif weekly_bull and daily_bull and vol_bull:
            desired_signal = BASE_SIZE * 0.9
        # Path 3: 1w bull + KAMA slope bull + RSI ok (fallback)
        elif weekly_bull and kama_bull and rsi_ok_long:
            desired_signal = BASE_SIZE * 0.8
        # Path 4: 1d bull + KAMA slope bull + volume (ensure trades)
        elif daily_bull and kama_bull and vol_bull:
            desired_signal = BASE_SIZE * 0.7
        
        # SHORT: Multiple confluence paths (any one triggers)
        # Path 1: Strong trend (1w + 1d + KAMA slope all bear) + RSI ok
        elif weekly_bear and daily_bear and kama_bear and rsi_ok_short:
            desired_signal = -BASE_SIZE
        # Path 2: 1w bear + 1d bear + volume confirm (looser)
        elif weekly_bear and daily_bear and vol_bear:
            desired_signal = -BASE_SIZE * 0.9
        # Path 3: 1w bear + KAMA slope bear + RSI ok (fallback)
        elif weekly_bear and kama_bear and rsi_ok_short:
            desired_signal = -BASE_SIZE * 0.8
        # Path 4: 1d bear + KAMA slope bear + volume (ensure trades)
        elif daily_bear and kama_bear and vol_bear:
            desired_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.65:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.45:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.65:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.45:
            final_signal = -BASE_SIZE * 0.6
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals