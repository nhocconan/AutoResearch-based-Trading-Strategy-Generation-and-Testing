#!/usr/bin/env python3
"""
Experiment #040: 4h Fisher Transform + 1d HMA Trend + Volume Breakout
Hypothesis: Ehlers Fisher Transform catches reversals better than RSI in bear/range markets.
Combined with 1d HMA for trend bias (only trade with HTF trend) and volume confirmation.
4h timeframe allows 20-50 trades/year (minimizes fee drag) while capturing swing moves.
Position sizing: 0.25 base, 0.35 max, discrete levels to reduce churn.
Stoploss: 2.5*ATR trailing stop to survive volatility spikes.
Key innovation: Fisher Transform + Volume spike detection + HTF trend alignment.
Entry conditions deliberately loose to ensure >=10 trades per symbol (critical requirement).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_1d_hma_vol_breakout_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals when Fisher crosses extreme levels (-2.0 / +2.0).
    Works well in bear/range markets where trend strategies fail.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    # Calculate median price (HL2)
    # For close-only, use close as proxy
    price = close.copy()
    
    # Normalize price to -1 to +1 range
    highest = pd.Series(price).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(price).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val[range_val == 0] = 0.001
    
    # Normalize: (price - lowest) / (highest - lowest) * 2 - 1
    normalized = np.zeros(n)
    normalized[:] = np.nan
    for i in range(period, n):
        if range_val[i] > 0:
            normalized[i] = (price[i] - lowest[i]) / range_val[i] * 2.0 - 1.0
            normalized[i] = np.clip(normalized[i], -0.999, 0.999)
    
    # Fisher Transform: 0.5 * ln((1 + normalized) / (1 - normalized))
    for i in range(period, n):
        if not np.isnan(normalized[i]):
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
            if i > period:
                trigger[i] = fisher[i - 1]
    
    return fisher, trigger

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

def calculate_volume_spike(volume, period=20):
    """Detect volume spikes (>1.5x average)."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_sma
    vol_ratio[np.isnan(vol_ratio)] = 1.0
    return vol_ratio

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(er_period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        if i == er_period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Standard RSI calculation."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    gains = np.zeros(n)
    losses = np.zeros(n)
    
    gains[1:] = np.where(delta > 0, delta, 0)
    losses[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    fisher, fisher_trigger = calculate_fisher_transform(close, period=9)
    atr = calculate_atr(high, low, close, 14)
    vol_ratio = calculate_volume_spike(volume, period=20)
    kama = calculate_kama(close, er_period=10)
    rsi = calculate_rsi(close, period=14)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Donchian channels for breakout detection
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    SIZE_HALF = 0.15
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF)
        bull_trend = close[i] > hma_1d_aligned[i]
        bear_trend = close[i] < hma_1d_aligned[i]
        
        # Fisher Transform signals (reversal detection)
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long_cross = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5 or fisher[i-1] <= -1.5)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short_cross = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5 or fisher[i-1] >= 1.5)
        
        # Fisher extreme levels (stronger signals)
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.3  # 30% above average
        
        # RSI confirmation (not too extreme)
        rsi_neutral = 30 < rsi[i] < 70
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        # Donchian breakout
        donchian_breakout_long = close[i] > donchian_high[i] * 0.998
        donchian_breakout_short = close[i] < donchian_low[i] * 1.002
        
        # EMA trend
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY (loose conditions to ensure trades) ===
        # Primary: Fisher extreme long + 1d bull trend
        if fisher_extreme_long and bull_trend:
            new_signal = SIZE_MAX
        # Secondary: Fisher cross long + volume spike + 1d bull trend
        elif fisher_long_cross and vol_spike and bull_trend:
            new_signal = SIZE_BASE
        # Tertiary: Fisher cross long + KAMA bullish + RSI oversold
        elif fisher_long_cross and kama_bullish and rsi_oversold:
            new_signal = SIZE_BASE
        # Fallback: Donchian breakout + 1d bull trend (ensures trades in strong trends)
        elif donchian_breakout_long and bull_trend and vol_spike:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY (loose conditions to ensure trades) ===
        # Primary: Fisher extreme short + 1d bear trend
        if fisher_extreme_short and bear_trend:
            new_signal = -SIZE_MAX
        # Secondary: Fisher cross short + volume spike + 1d bear trend
        elif fisher_short_cross and vol_spike and bear_trend:
            new_signal = -SIZE_BASE
        # Tertiary: Fisher cross short + KAMA bearish + RSI overbought
        elif fisher_short_cross and kama_bearish and rsi_overbought:
            new_signal = -SIZE_BASE
        # Fallback: Donchian breakout + 1d bear trend (ensures trades in strong trends)
        elif donchian_breakout_short and bear_trend and vol_spike:
            new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 4h timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals