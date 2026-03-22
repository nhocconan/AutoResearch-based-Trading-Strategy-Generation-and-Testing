#!/usr/bin/env python3
"""
Experiment #097: 15m Supertrend + 4h HMA Trend + 1h RSI Momentum + BB Squeeze Regime
Hypothesis: 15m timeframe needs strong HTF filters to avoid noise whipsaws.
4h HMA provides stable trend bias (proven in #088 with Sharpe=0.223).
1h RSI adds momentum confirmation between 4h trend and 15m entries.
Bollinger Band squeeze detection identifies low-volatility breakout setups.
Volume confirmation filters false breakouts (critical on 15m).

Why this might work on 15m (learning from #085 Sharpe=-4.051 failure):
- #085 failed because pure RSI mean-reversion on 15m gets killed by trends
- This strategy is TREND-FOLLOWING with HTF confirmation (not mean-reversion)
- 4h HMA filter ensures we only trade in direction of higher timeframe trend
- 1h RSI momentum (not extremes) confirms entry timing
- BB squeeze + volume spike = genuine breakout, not noise
- ATR stoploss at 2.0x protects against 15m whipsaws
- Discrete position sizing (0.25/0.35) minimizes fee churn

Timeframe: 15m (REQUIRED), HTF: 4h and 1h via mtf_data helper (call ONCE before loop).
Position sizing: 0.25 base, 0.35 strong signals. Stoploss at 2.0*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_supertrend_4h_hma_1h_rsi_bb_squeeze_vol_v1"
timeframe = "15m"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    # Bandwidth = (upper - lower) / sma
    bandwidth = np.zeros(len(close))
    mask = sma > 0
    bandwidth[mask] = (upper[mask] - lower[mask]) / sma[mask]
    return upper, lower, bandwidth, sma

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator.
    Returns: supertrend_values, supertrend_direction (1=long, -1=short)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
            
        if direction[i-1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            if close[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            if close[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
    
    return supertrend, direction

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate 4h HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Calculate 1h HTF indicators
    rsi_1h = calculate_rsi(df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    volume_ma = calculate_volume_ma(volume, 20)
    
    # Supertrend (10, 3) - proven parameters
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    
    # EMA for trend confirmation
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    
    # Calculate BB bandwidth percentile for squeeze detection
    # Squeeze = bandwidth in bottom 20% of last 100 bars
    bb_percentile = np.zeros(n)
    for i in range(100, n):
        if not np.isnan(bb_bandwidth[i]):
            recent_bw = bb_bandwidth[i-100:i+1]
            recent_bw = recent_bw[~np.isnan(recent_bw)]
            if len(recent_bw) > 0:
                bb_percentile[i] = np.sum(recent_bw <= bb_bandwidth[i]) / len(recent_bw)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]) or np.isnan(supertrend[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HTF TREND BIAS (most important filter) ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === 1H RSI MOMENTUM (entry timing) ===
        # RSI > 50 = bullish momentum, RSI < 50 = bearish momentum
        rsi_momentum_long = rsi_1h_aligned[i] > 50
        rsi_momentum_short = rsi_1h_aligned[i] < 50
        
        # RSI not at extremes (avoid reversal zones)
        rsi_not_overbought = rsi_1h_aligned[i] < 75
        rsi_not_oversold = rsi_1h_aligned[i] > 25
        
        # === SUPERTREND SIGNAL (15m entry trigger) ===
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        
        # === EMA ALIGNMENT (trend confirmation) ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === BB SQUEEZE REGIME (breakout setup) ===
        # bb_percentile < 0.25 = bandwidth in bottom 25% = squeeze
        bb_squeeze = bb_percentile[i] < 0.25
        
        # === VOLUME CONFIRMATION (filter false breakouts) ===
        volume_spike = volume[i] > 1.5 * volume_ma[i] if not np.isnan(volume_ma[i]) else False
        
        new_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Path 1: Strong signal - all filters align (4h trend + 1h RSI + Supertrend + EMA + volume)
        if bull_trend_4h and rsi_momentum_long and rsi_not_overbought and st_bullish and ema_bullish:
            if volume_spike or bb_squeeze:
                new_signal = SIZE_STRONG
            else:
                new_signal = SIZE_BASE
        
        # Path 2: Simpler entry - 4h trend + Supertrend + RSI momentum (ensures trades)
        if new_signal == 0.0 and bull_trend_4h and st_bullish and rsi_momentum_long:
            if ema_bullish or bb_squeeze:
                new_signal = SIZE_BASE
        
        # Path 3: Fallback - 4h trend + Supertrend only (ensures minimum trades)
        if new_signal == 0.0 and bull_trend_4h and st_bullish:
            if rsi_momentum_long or ema_bullish:
                new_signal = SIZE_BASE
        
        # === SHORT ENTRY CONDITIONS ===
        # Path 1: Strong signal - all filters align
        if bear_trend_4h and rsi_momentum_short and rsi_not_oversold and st_bearish and ema_bearish:
            if volume_spike or bb_squeeze:
                new_signal = -SIZE_STRONG
            else:
                new_signal = -SIZE_BASE
        
        # Path 2: Simpler entry - 4h trend + Supertrend + RSI momentum
        if new_signal == 0.0 and bear_trend_4h and st_bearish and rsi_momentum_short:
            if ema_bearish or bb_squeeze:
                new_signal = -SIZE_BASE
        
        # Path 3: Fallback - 4h trend + Supertrend only
        if new_signal == 0.0 and bear_trend_4h and st_bearish:
            if rsi_momentum_short or ema_bearish:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR for 15m ===
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            stoploss_price = highest_close - 2.0 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            stoploss_price = lowest_close + 2.0 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals