#!/usr/bin/env python3
"""
Experiment #008: 30m Regime-Adaptive Strategy with 4h Trend Filter
Hypothesis: 30m timeframe captures intraday swings while 4h HTF provides trend direction.
Use Bollinger Band Width percentile to detect regime: squeeze=range, expansion=trend.
In trend regime: follow 4h HMA direction with 30m RSI pullback entries.
In range regime: mean reversion with RSI extremes (buy <35, sell >65).
ATR-based stoploss (2.5*ATR) protects against crashes. Discrete sizing (0.0, ±0.25, ±0.30).
This should work in both 2021-2024 bull/bear and 2025 range markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_regime_adaptive_30m_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bw = (upper - lower) / sma
    bw_pct = pd.Series(bw).rolling(window=100, min_periods=100).rank(pct=True).values
    return upper, lower, bw, bw_pct

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_supertrend(high, low, close, period=10, mult=3.0):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    
    supertrend = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish
    
    for i in range(period, len(close)):
        if close[i] > supertrend[i-1] if i > 0 else lower[i]:
            supertrend[i] = lower[i]
            direction[i] = 1
        else:
            supertrend[i] = upper[i]
            direction[i] = -1
    
    return supertrend, direction

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_fast = calculate_ema(close, 8)
    ema_slow = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    bb_upper, bb_lower, bb_width, bb_width_pct = calculate_bollinger_bands(close, 20, 2.0)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=0.0)
    
    signals = np.zeros(n)
    SIZE = 0.30
    HALF_SIZE = 0.15
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(100, n):
        # Regime detection using Bollinger Band Width percentile
        # BBW pct < 0.3 = squeeze (range), BBW pct > 0.7 = expansion (trend)
        in_range_regime = bb_width_pct[i] < 0.35 if not np.isnan(bb_width_pct[i]) else False
        in_trend_regime = bb_width_pct[i] > 0.65 if not np.isnan(bb_width_pct[i]) else False
        
        # 4h trend filter
        hma_4h_bullish = hma_4h_aligned[i] > 0 and close[i] > hma_4h_aligned[i]
        hma_4h_bearish = hma_4h_aligned[i] > 0 and close[i] < hma_4h_aligned[i]
        
        # Supertrend direction
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # EMA alignment
        ema_bullish = ema_fast[i] > ema_slow[i] and close[i] > ema_50[i]
        ema_bearish = ema_fast[i] < ema_slow[i] and close[i] < ema_50[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral_long = 45 < rsi[i] < 65
        rsi_neutral_short = 35 < rsi[i] < 55
        
        # Volume confirmation
        vol_ok = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        new_signal = 0.0
        
        # === TREND REGIME: Follow 4h direction with pullback entries ===
        if in_trend_regime:
            # Long: 4h bullish + pullback (RSI not overbought) + EMA aligned
            if hma_4h_bullish and rsi_neutral_long and ema_bullish and vol_ok:
                new_signal = SIZE
            # Short: 4h bearish + bounce (RSI not oversold) + EMA aligned
            elif hma_4h_bearish and rsi_neutral_short and ema_bearish and vol_ok:
                new_signal = -SIZE
        
        # === RANGE REGIME: Mean reversion with RSI extremes ===
        elif in_range_regime:
            # Long: RSI oversold + price near lower BB
            if rsi_oversold and close[i] < bb_lower[i] * 1.002:
                new_signal = SIZE
            # Short: RSI overbought + price near upper BB
            elif rsi_overbought and close[i] > bb_upper[i] * 0.998:
                new_signal = -SIZE
        
        # === DEFAULT: Supertrend + EMA filter (catches remaining opportunities) ===
        else:
            # Long: Supertrend bullish + EMA bullish + RSI ok
            if st_bullish and ema_bullish and rsi[i] > 40 and rsi[i] < 70:
                new_signal = SIZE
            # Short: Supertrend bearish + EMA bearish + RSI ok
            elif st_bearish and ema_bearish and rsi[i] > 30 and rsi[i] < 60:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop: reduce to half at 3R profit
            elif close[i] > entry_price[i-1] + 3.0 * atr[i]:
                if signals[i-1] == SIZE and new_signal == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop: reduce to half at 3R profit
            elif close[i] < entry_price[i-1] - 3.0 * atr[i]:
                if signals[i-1] == -SIZE and new_signal == -SIZE:
                    new_signal = -HALF_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0 and position_side == 0:
            # New position
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                # Reversal
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
            highest_since_entry[i] = max(highest_since_entry[i-1], close[i])
            lowest_since_entry[i] = min(lowest_since_entry[i-1], close[i])
        else:
            # Position closed or flat
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else close[i]
            lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else close[i]
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals