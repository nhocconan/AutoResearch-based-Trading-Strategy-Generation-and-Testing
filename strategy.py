#!/usr/bin/env python3
"""
Experiment #025: 15m Multi-Timeframe Pullback with 4h HMA Regime Filter
Hypothesis: 15m captures short-term pullbacks within 4h HMA-defined trend regime.
Key insight: 15m needs STRONG HTF filter to avoid whipsaws (seen in exp#013, #019 failures).
Using proven 4h HMA regime (from best strategies) + 15m RSI pullback entries + ATR stops.
Position sizing: 0.20-0.30 discrete levels, 2.5*ATR trailing stoploss.
Why this might work: 4h HMA filters 2022 crash regime, 15m entries generate sufficient trades.
Timeframe: 15m (REQUIRED for exp#025), HTF: 4h via mtf_data helper.
Must generate 10+ trades on train, 3+ on test - entry conditions LOOSENED vs failed experiments.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_pullback_4h_hma_rsi_atr_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(close, period):
    """Calculate SMA."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_keltner_channel(high, low, close, period=20, atr_mult=2.0):
    """Calculate Keltner Channel for volatility-based support/resistance."""
    ema_mid = calculate_ema(close, period)
    atr = calculate_atr(high, low, close, period)
    upper = ema_mid + atr_mult * atr
    lower = ema_mid - atr_mult * atr
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for regime filter
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    sma_50 = calculate_sma(close, 50)
    
    # Keltner channels for mean reversion levels
    kc_upper, kc_lower = calculate_keltner_channel(high, low, close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 15m trend confirmation
        bull_trend_15m = ema_21[i] > ema_50[i]
        bear_trend_15m = ema_21[i] < ema_50[i]
        
        # Long-term filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        # RSI conditions - LOOSENED for more trades (critical for 10+ trades requirement)
        rsi_pullback_long = 25 < rsi[i] < 55
        rsi_bounce_short = 45 < rsi[i] < 75
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Keltner channel mean reversion
        near_kc_lower = close[i] <= kc_lower[i] * 1.005
        near_kc_upper = close[i] >= kc_upper[i] * 0.995
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.02 and close[i] >= ema_21[i] * 0.98
        price_near_ema21_short = close[i] >= ema_21[i] * 0.98 and close[i] <= ema_21[i] * 1.02
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 5:
            higher_low = low[i] > min(low[i-3:i])
            lower_high = high[i] < max(high[i-3:i])
        
        # EMA alignment
        ema_aligned_long = ema_21[i] > ema_50[i] and ema_50[i] > ema_200[i] if not np.isnan(ema_200[i]) else ema_21[i] > ema_50[i]
        ema_aligned_short = ema_21[i] < ema_50[i] and ema_50[i] < ema_200[i] if not np.isnan(ema_200[i]) else ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === LONG ENTRIES (only when 4h bullish) ===
        if bull_trend_4h:
            # Primary: Pullback to EMA21 with RSI confirmation
            if price_near_ema21_long and rsi_pullback_long:
                new_signal = SIZE_BASE
            
            # Secondary: RSI oversold bounce in uptrend
            elif rsi_oversold and bull_trend_15m:
                new_signal = SIZE_HALF
            
            # Tertiary: Keltner lower touch in uptrend
            elif near_kc_lower and bull_trend_4h and rsi[i] > 30:
                new_signal = SIZE_HALF
            
            # Momentum: Higher low with trend
            elif higher_low and ema_aligned_long and rsi[i] > 40:
                new_signal = SIZE_HALF
        
        # === SHORT ENTRIES (only when 4h bearish) ===
        elif bear_trend_4h:
            # Primary: Bounce to EMA21 with RSI confirmation
            if price_near_ema21_short and rsi_bounce_short:
                new_signal = -SIZE_BASE
            
            # Secondary: RSI overbought rejection in downtrend
            elif rsi_overbought and bear_trend_15m:
                new_signal = -SIZE_HALF
            
            # Tertiary: Keltner upper touch in downtrend
            elif near_kc_upper and bear_trend_4h and rsi[i] < 70:
                new_signal = -SIZE_HALF
            
            # Momentum: Lower high with trend
            elif lower_high and ema_aligned_short and rsi[i] < 60:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            # New position
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            # Position reversal
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            # Position closed
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals