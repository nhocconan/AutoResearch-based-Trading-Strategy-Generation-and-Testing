#!/usr/bin/env python3
"""
Experiment #023: 12h Asymmetric Trend with 1d/1w HMA Regime Filter
Hypothesis: Asymmetric logic (long in bull, short only in strong bear) reduces whipsaws.
12h timeframe = ~2 bars/day = fewer trades but higher quality vs 1h/4h.
Key improvements over failed exp#005/exp#017:
- OR logic for HTF (1d OR 1w bullish) = more trades than AND logic
- Looser RSI entry (30-60 long, 40-70 short) = ensures 10+ trades
- Volume spike filter only on breakouts, not pullbacks = more entries
- Simpler stoploss: fixed 2.5*ATR from entry, not complex trailing
- Position sizing: 0.25 base, 0.30 strong trend (discrete levels)
Timeframe: 12h (REQUIRED), HTF: 1d + 1w via mtf_data helper
Must work on BTC/ETH/SOL individually - no SOL-only bias
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_asym_dual_htm_rsi_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """RSI calculation."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(len(close))
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi = np.zeros(len(close))
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_ema(close, period):
    """EMA calculation."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """SMA calculation."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_ma(volume, period=20):
    """Volume moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF HMAs
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (CRITICAL - Rule 2, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    sma_200 = calculate_sma(close, 200)
    vol_ma = calculate_volume_ma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    stoploss_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            signals[i] = 0.0
            continue
        
        # HTF trend bias - OR logic for more trades (1d OR 1w)
        bull_1d = close[i] > hma_1d_aligned[i]
        bull_1w = close[i] > hma_1w_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        bear_1w = close[i] < hma_1w_aligned[i]
        
        # Bull regime: at least one HTF bullish
        bull_regime = bull_1d or bull_1w
        # Bear regime: BOTH HTF bearish (stricter for shorts - asymmetric)
        bear_regime = bear_1d and bear_1w
        
        # 12h trend confirmation
        bull_12h = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_12h = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term filter
        above_200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # Volume confirmation
        vol_above_avg = volume[i] > vol_ma[i] * 0.9
        
        # RSI conditions - LOOSENED for more trades
        rsi_long_ok = 30 < rsi[i] < 60
        rsi_short_ok = 40 < rsi[i] < 70
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Price pullback to EMA21
        pullback_long = close[i] <= ema_21[i] * 1.02 and close[i] >= ema_21[i] * 0.96
        bounce_short = close[i] >= ema_21[i] * 0.98 and close[i] <= ema_21[i] * 1.04
        
        # Price action
        higher_low = False
        lower_high = False
        if i >= 3:
            higher_low = low[i] > low[i-3]
            lower_high = high[i] < high[i-3]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (asymmetric: easier to enter long) ===
        if bull_regime:
            # Primary: Pullback to EMA21 with RSI confirmation
            if pullback_long and rsi_long_ok:
                new_signal = SIZE_BASE
            
            # Secondary: RSI oversold bounce in bull regime
            elif rsi_oversold and bull_12h:
                new_signal = SIZE_BASE
            
            # Tertiary: Higher low with volume
            elif higher_low and bull_12h and vol_above_avg:
                new_signal = SIZE_BASE
            
            # Strong trend: All aligned
            if bull_regime and bull_12h and above_200 and rsi[i] > 45:
                new_signal = SIZE_STRONG
        
        # === SHORT ENTRIES (asymmetric: stricter, both HTF bearish) ===
        if bear_regime:
            # Primary: Bounce to EMA21 with RSI confirmation
            if bounce_short and rsi_short_ok:
                new_signal = -SIZE_BASE
            
            # Secondary: RSI overbought rejection in bear regime
            elif rsi_overbought and bear_12h:
                new_signal = -SIZE_BASE
            
            # Tertiary: Lower high with volume
            elif lower_high and bear_12h and vol_above_avg:
                new_signal = -SIZE_BASE
            
            # Strong trend: All aligned
            if bear_regime and bear_12h and below_200 and rsi[i] < 55:
                new_signal = -SIZE_STRONG
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Check stoploss for existing positions
        if position_side > 0 and entry_price > 0:
            if close[i] < stoploss_price:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            if close[i] > stoploss_price:
                new_signal = 0.0
        
        # Update position tracking
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stoploss_price = entry_price - 2.5 * atr[i] if position_side > 0 else entry_price + 2.5 * atr[i]
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            stoploss_price = entry_price - 2.5 * atr[i] if position_side > 0 else entry_price + 2.5 * atr[i]
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            stoploss_price = 0.0
        
        signals[i] = new_signal
    
    return signals