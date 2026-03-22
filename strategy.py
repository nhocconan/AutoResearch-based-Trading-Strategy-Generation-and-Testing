#!/usr/bin/env python3
"""
Experiment #004: 4h Primary + 12h/1d HTF — Adaptive KAMA + ADX Regime

Hypothesis: Previous Connors RSI strategies failed because they're too mean-reversion
focused for crypto's trending nature. This strategy uses:

1. KAMA (Kaufman Adaptive MA) - adapts speed to market efficiency ratio
   Faster in trends, slower in chop. Better than HMA for crypto volatility.
2. ADX(14) for trend strength - only trade when ADX > 20 (trending) or < 25 (range)
3. Bollinger Band Width percentile - detect squeeze vs expansion
4. RSI(14) for entry timing - less extreme than Connors, more reliable
5. 12h KAMA for major trend bias, 1d KAMA for macro filter
6. ATR(14) trailing stop at 2.5x

Why this differs from failed experiments:
- KAMA adapts to volatility (HMA doesn't)
- ADX regime filter (not Choppiness Index)
- RSI(14) not Connors RSI (simpler, more robust)
- BB Width for volatility regime (not CHOP)
- Asymmetric sizing: 0.30 in strong trends, 0.20 in weak trends

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h and 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.20-0.30 discrete based on ADX strength
Stoploss: 2.5 * ATR(14) trailing
Target trades: 20-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_bb_regime_12h1d_v1"
timeframe = "4h"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index) for trend strength.
    ADX > 25 = strong trend, ADX < 20 = range/chop
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio.
    Fast in trends, slow in chop.
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio (ER) = |net change| / sum of absolute changes
    change = np.abs(close_s.diff().values)
    net_change = np.abs(close_s.diff().values)
    
    # Sum of absolute price changes over er_period
    sum_changes = pd.Series(change).rolling(window=er_period, min_periods=er_period).sum().values
    net_changes = pd.Series(net_change).rolling(window=er_period, min_periods=er_period).apply(
        lambda x: np.abs(x.sum()) if len(x) >= er_period else 0
    ).values
    
    # Avoid division by zero
    er = np.zeros(n)
    mask = sum_changes > 0
    er[mask] = net_changes[mask] / sum_changes[mask]
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma * 100
    
    return upper.values, lower.values, bandwidth.values

def calculate_bb_percentile(bandwidth, lookback=100):
    """Calculate bandwidth percentile over lookback period."""
    bw_s = pd.Series(bandwidth)
    bb_pct = bw_s.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= lookback else 50
    )
    return bb_pct.fillna(50).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    kama_12h_21 = calculate_kama(df_12h['close'].values, 10, 2, 30)
    kama_1d_21 = calculate_kama(df_1d['close'].values, 10, 2, 30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_21)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    kama_4h_21 = calculate_kama(close, 10, 2, 30)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    bb_pct = calculate_bb_percentile(bb_bandwidth, 100)
    
    # Volume SMA for filter
    volume_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE_STRONG = 0.30  # ADX > 30 (strong trend)
    BASE_SIZE_WEAK = 0.20    # ADX 20-30 (weak trend)
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(kama_4h_21[i]) or np.isnan(bb_pct[i]):
            continue
        
        if np.isnan(volume_sma[i]) or volume_sma[i] == 0:
            continue
        
        # === 1D TREND BIAS (MACRO) ===
        # Price above 1d KAMA = bullish macro bias
        # Price below 1d KAMA = bearish macro bias
        trend_1d_bullish = close[i] > kama_1d_aligned[i]
        trend_1d_bearish = close[i] < kama_1d_aligned[i]
        
        # === 12H TREND CONFIRMATION (INTERMEDIATE) ===
        trend_12h_bullish = close[i] > kama_12h_aligned[i]
        trend_12h_bearish = close[i] < kama_12h_aligned[i]
        
        # === 4H KAMA TREND ===
        trend_4h_bullish = close[i] > kama_4h_21[i]
        trend_4h_bearish = close[i] < kama_4h_21[i]
        
        # === ADX TREND STRENGTH ===
        # ADX > 30 = strong trend (use larger size)
        # ADX 20-30 = moderate trend (use smaller size)
        # ADX < 20 = range (mean reversion mode)
        adx_strong = adx_14[i] > 30
        adx_moderate = 20 <= adx_14[i] <= 30
        adx_weak = adx_14[i] < 20
        
        # === BOLLINGER BAND VOLATILITY REGIME ===
        # BB% < 20 = squeeze (low vol, breakout imminent)
        # BB% > 80 = expansion (high vol, potential reversal)
        bb_squeeze = bb_pct[i] < 20
        bb_expansion = bb_pct[i] > 80
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * volume_sma[i]
        
        # === RSI ENTRY SIGNALS ===
        # RSI < 35 = oversold (long opportunity in uptrend)
        # RSI > 65 = overbought (short opportunity in downtrend)
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING BASED ON ADX ===
        if adx_strong:
            current_size = BASE_SIZE_STRONG
        elif adx_moderate:
            current_size = BASE_SIZE_WEAK
        else:
            current_size = BASE_SIZE_WEAK * 0.7  # Reduce size in range
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: 1d bullish OR 12h bullish (macro bias)
        # Plus: 4h KAMA bullish + RSI oversold + volume
        # In squeeze: more aggressive (breakout play)
        if trend_1d_bullish or trend_12h_bullish:
            if trend_4h_bullish and rsi_oversold and volume_ok:
                if bb_squeeze:
                    new_signal = current_size * 1.2  # Boost on squeeze breakout
                else:
                    new_signal = current_size
        
        # SHORT ENTRIES
        # Require: 1d bearish OR 12h bearish (macro bias)
        # Plus: 4h KAMA bearish + RSI overbought + volume
        if trend_1d_bearish or trend_12h_bearish:
            if trend_4h_bearish and rsi_overbought and volume_ok:
                if bb_squeeze:
                    new_signal = -current_size * 1.2  # Boost on squeeze breakout
                else:
                    new_signal = -current_size
        
        # === RANGE MODE MEAN REVERSION (ADX < 20) ===
        # In range: fade extremes at BB bounds
        if adx_weak and not in_position:
            if bb_expansion and rsi_oversold and close[i] < bb_lower[i]:
                new_signal = current_size * 0.8  # Smaller size for mean reversion
            elif bb_expansion and rsi_overbought and close[i] > bb_upper[i]:
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 300 bars (~50 days on 4h), allow weaker entry
        if bars_since_last_trade > 300 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and trend_12h_bullish and rsi_14[i] < 40:
                new_signal = current_size * 0.5
            elif trend_1d_bearish and trend_12h_bearish and rsi_14[i] > 60:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0:
                # Long position: exit if 12h trend flips bearish
                if trend_12h_bearish and rsi_14[i] > 60:
                    trend_reversal = True
            if position_side < 0:
                # Short position: exit if 12h trend flips bullish
                if trend_12h_bullish and rsi_14[i] < 40:
                    trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals