#!/usr/bin/env python3
"""
Experiment #056: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume Filter

Hypothesis: After 55 failed experiments, the critical lesson for 30m timeframe:
- Too many strict filters = 0 trades (experiments #045, #049, #050, #052, #053 all Sharpe=0.000)
- Session filters often block too many entries on crypto (24/7 market)
- Pure trend following fails on BTC/ETH bear markets (2022 crash, 2025 bear)
- SOLUTION: Dual HTF bias (4h + 1d HMA) + RSI pullback entries (NOT breakouts)
- Remove session filter (crypto trades 24/7, volume spikes anytime)
- LOOSE RSI zones (30-70) to ensure trades generate on all symbols
- Volume ratio adds conviction but not required

Key design choices:
- Timeframe: 30m (target 40-80 trades/year, NOT >100)
- HTF: 4h HMA(21) for medium trend, 1d HMA(50) for major trend
- Entry: RSI(14) pullback to 30-70 zone in direction of HTF trend
- Volume: taker_buy_volume ratio for confirmation (optional)
- Position size: 0.20 (20% of capital, conservative for 30m)
- Stoploss: 2.5x ATR(14) trailing
- LOOSE filters to ensure >=40 trades/year on EACH symbol

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=40/year on train, trades>=3 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_volume_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    hma_30m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (taker buy pressure)
    volume_ratio = np.zeros(n)
    volume_ratio[:] = np.nan
    for i in range(n):
        if volume[i] > 1e-10:
            volume_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            volume_ratio[i] = 0.5
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 30m)
    
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
        if np.isnan(hma_30m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h + 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias: both HTF agree
        strong_bull = htf_4h_bull and htf_1d_bull
        strong_bear = htf_4h_bear and htf_1d_bear
        
        # === 30m HMA TREND ===
        hma_bull = close[i] > hma_30m[i]
        hma_bear = close[i] < hma_30m[i]
        
        # === RSI PULLBACK ZONES (LOOSE - ensure trades generate) ===
        # Long: RSI pulled back but not oversold (30-60)
        rsi_ok_long = 30.0 <= rsi[i] <= 60.0
        # Short: RSI pulled back but not overbought (40-70)
        rsi_ok_short = 40.0 <= rsi[i] <= 70.0
        
        # === VOLUME CONFIRMATION (OPTIONAL) ===
        volume_bull = volume_ratio[i] > 0.45
        volume_bear = volume_ratio[i] < 0.55
        
        # === DESIRED SIGNAL (LOOSE CONDITIONS TO ENSURE TRADES) ===
        desired_signal = 0.0
        
        # LONG ENTRY: Strong HTF bull + RSI pullback (volume optional)
        if strong_bull and rsi_ok_long:
            desired_signal = SIZE
        # Fallback long: 4h bull + 30m bull + RSI not overbought
        elif htf_4h_bull and hma_bull and rsi[i] < 65.0:
            desired_signal = SIZE * 0.7
        # Even looser: RSI very oversold in any uptrend
        elif rsi[i] < 35.0 and htf_4h_bull:
            desired_signal = SIZE * 0.7
        
        # SHORT ENTRY: Strong HTF bear + RSI pullback (volume optional)
        elif strong_bear and rsi_ok_short:
            desired_signal = -SIZE
        # Fallback short: 4h bear + 30m bear + RSI not oversold
        elif htf_4h_bear and hma_bear and rsi[i] > 35.0:
            desired_signal = -SIZE * 0.7
        # Even looser: RSI very overbought in any downtrend
        elif rsi[i] > 65.0 and htf_4h_bear:
            desired_signal = -SIZE * 0.7
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
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