# Strategy: mtf_4h_hma_rsi_pullback_12h_adx_atr_v1

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| BTCUSDT | -0.681 | -9.9% | -22.1% | 483 | FAIL |
| ETHUSDT | -0.131 | +9.8% | -28.2% | 510 | FAIL |
| SOLUSDT | 0.634 | +86.6% | -20.9% | 529 | PASS |

## Test Results (2025+)
| Symbol | Sharpe | Return | Max DD | Trades | Status |
|--------|--------|--------|--------|--------|--------|
| SOLUSDT | 0.568 | +15.8% | -9.3% | 163 | PASS |

## Code
```python
#!/usr/bin/env python3
"""
Experiment #1084: 4h Primary + 12h/1d HTF — HMA Trend + RSI Pullback + ADX Filter

Hypothesis: After 785+ failed experiments, the winning pattern simplifies to:
1. HMA (Hull Moving Average) — faster response than EMA, less lag than SMA
   HMA(21) vs HMA(48) crossover for trend direction on 4h
2. 12h HMA21 for macro bias — only trade in direction of higher TF trend
3. RSI(14) pullback entries — not extremes, but dips within trend (RSI 40-50 in uptrend)
   This generates MORE trades than CRSI extremes while maintaining quality
4. ADX(14) > 20 filter — avoid dead chop where trend strategies fail
5. ATR(14) trailing stop 2.5x — proper risk management

Why this should beat Sharpe=0.612:
- HMA is proven faster/smoother than EMA (used in winning baseline mtf_hma_rsi_zscore_v1)
- RSI pullback (not extremes) generates 30-60 trades/year vs CRSI's 0-10
- 12h HTF filter prevents counter-trend trades (major failure mode in 2022)
- ADX filter avoids choppy whipsaws
- Simpler = more robust across BTC/ETH/SOL

Timeframe: 4h (primary)
HTF: 12h — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_12h_adx_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average — faster and smoother than EMA.
    
    Formula:
    1. WMA(period/2) * 2
    2. WMA(period) * 1
    3. Diff = (1) - (2)
    4. HMA = WMA(sqrt(period)) of Diff
    """
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """
    Relative Strength Index — momentum oscillator.
    
    Formula:
    1. Calculate gains and losses
    2. EMA of gains and losses over period
    3. RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    # Pad first element
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0  # When no loss, RSI = 50
    
    return rsi

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — trend strength indicator.
    
    Formula:
    1. Calculate +DM and -DM
    2. Calculate TR (True Range)
    3. Smooth +DM, -DM, TR over period
    4. +DI = +DM / TR, -DI = -DM / TR
    5. DX = 100 * |+DI - -DI| / (+DI + -DI)
    6. ADX = EMA of DX over period
    
    ADX > 25 = strong trend
    ADX < 20 = weak/choppy market
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = close[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with EMA
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = tr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX
    di_sum = plus_di + minus_di
    dx = np.zeros(n)
    mask2 = di_sum > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    # ADX = EMA of DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
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
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA21 for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    hma_21 = calculate_hma(close, 21)
    hma_48 = calculate_hma(close, 48)
    rsi = calculate_rsi(close, period=14)
    adx = calculate_adx(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_21[i]) or np.isnan(hma_48[i]):
            continue
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (12h HMA21) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === PRIMARY TREND (4h HMA crossover) ===
        hma_bull = hma_21[i] > hma_48[i]
        hma_bear = hma_21[i] < hma_48[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx[i] > 20.0
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back to 40-50 in uptrend (not oversold extreme)
        rsi_pullback_long = 40.0 <= rsi[i] <= 55.0
        # Short: RSI rallied to 45-60 in downtrend (not overbought extreme)
        rsi_pullback_short = 45.0 <= rsi[i] <= 60.0
        
        # === VOLATILITY CHECK ===
        vol_spike = atr[i] > 2.0 * np.nanmedian(atr[max(0, i-100):i]) if i > 100 else False
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # All conditions must align: macro bull + HMA bull + ADX strong + RSI pullback
        if macro_bull and hma_bull and strong_trend and rsi_pullback_long:
            desired_signal = current_size
        
        # === SHORT ENTRY ===
        # All conditions must align: macro bear + HMA bear + ADX strong + RSI pullback
        elif macro_bear and hma_bear and strong_trend and rsi_pullback_short:
            desired_signal = -current_size
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HMA still bullish or RSI not overbought
                if hma_bull and rsi[i] < 70.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if HMA still bearish or RSI not oversold
                if hma_bear and rsi[i] > 30.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if HMA crosses bearish or macro reverses
            if hma_bear and rsi[i] > 65.0:
                desired_signal = 0.0
            if macro_bear and adx[i] > 25.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if HMA crosses bullish or macro reverses
            if hma_bull and rsi[i] < 35.0:
                desired_signal = 0.0
            if macro_bull and adx[i] > 25.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals
```

## Last Updated
2026-03-23 19:33
