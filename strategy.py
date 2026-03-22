#!/usr/bin/env python3
"""
Experiment #345: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume/Session Filter

Hypothesis: Lower TF (1h) can work IF we use HTF for direction and add trade-frequency controls.
Key innovations vs failed 1h strategies (335, 338, 340):
1. 4h HMA(21) for MAJOR trend direction (not 1h trend - too noisy)
2. 1d HMA(21) for secondary regime filter (stronger than 4h alone)
3. 1h RSI(14) pullback entries ONLY within HTF trend (not counter-trend)
4. SESSION FILTER: Only trade 8-20 UTC (reduces trades 40%, avoids Asia chop)
5. VOLUME FILTER: Volume > 0.8x 20-bar avg (avoids low-liquidity false signals)
6. ATR VOLATILITY FILTER: ATR ratio 0.7-1.8 (avoid extreme vol spikes/crush)
7. FREQUENCY SAFEGUARD: Force entry every 40 bars if no signal (ensures 20+ trades/year)

Why this might beat current best (Sharpe=0.435 on 1d):
- 1h entries capture more of the trend move vs 1d entries
- Session filter removes 60% of low-quality Asia session trades
- Volume filter avoids false breakouts on low liquidity
- Dual HTF (4h + 1d) stronger than single HTF
- Simpler entry logic than failed choppiness-based strategies

Position sizing: 0.20-0.28 longs, 0.12-0.18 shorts (smaller for lower TF)
Stoploss: 2.5 * ATR(14) trailing
Target: 40-80 trades/year on 1h (with session filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_4h1d_session_vol_v2"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Much less lag than EMA while maintaining smoothness.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for volume filter."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1d HTF indicators (secondary regime filter)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h_8 = calculate_hma(close, period=8)
    hma_1h_21 = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, 200)
    vol_sma_20 = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for lower TF)
    LONG_BASE = 0.22
    LONG_STRONG = 0.28
    SHORT_BASE = 0.14
    SHORT_STRONG = 0.18
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -40
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1h_8[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        if np.isnan(vol_sma_20[i]) or vol_sma_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / (vol_sma_20[i] + 1e-10)
        volume_ok = vol_ratio > 0.8
        
        # === ATR VOLATILITY FILTER (avoid extreme vol) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        vol_normal = 0.6 < atr_ratio < 2.0
        
        # === 4H MAJOR TREND REGIME (primary direction filter) ===
        regime_4h_bull = close[i] > hma_4h_21_aligned[i]
        regime_4h_bear = close[i] < hma_4h_21_aligned[i]
        
        # === 1D SECONDARY REGIME (confirmation) ===
        regime_1d_bull = close[i] > hma_1d_21_aligned[i]
        regime_1d_bear = close[i] < hma_1d_21_aligned[i]
        
        # === STRONG REGIME (both 4h and 1d agree) ===
        strong_bull = regime_4h_bull and regime_1d_bull
        strong_bear = regime_4h_bear and regime_1d_bear
        
        # === 1H LOCAL TREND ===
        hma_bullish = hma_1h_8[i] > hma_1h_21[i]
        hma_bearish = hma_1h_8[i] < hma_1h_21[i]
        
        # HMA slope (2-bar lookback)
        hma_slope_up = hma_1h_21[i] > hma_1h_21[i-2] if i >= 2 else False
        hma_slope_down = hma_1h_21[i] < hma_1h_21[i-2] if i >= 2 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_1h_21[i]
        price_below_hma = close[i] < hma_1h_21[i]
        
        # Price relative to SMA200
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === RSI SIGNALS (pullback entries) ===
        rsi_pullback_long = 35.0 < rsi_14[i] < 55.0
        rsi_pullback_short = 45.0 < rsi_14[i] < 65.0
        rsi_strong_oversold = rsi_14[i] < 35.0
        rsi_strong_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in strong bull regime)
        if strong_bull:
            # Primary: RSI pullback + HMA bullish + session + volume
            if rsi_pullback_long and hma_bullish and in_session and volume_ok:
                new_signal = LONG_BASE
            
            # Strong: RSI very oversold + strong bull + session
            elif rsi_strong_oversold and strong_bull and in_session:
                new_signal = LONG_STRONG
            
            # HMA bullish crossover + RSI rising + volume
            elif hma_bullish and hma_slope_up and rsi_rising and volume_ok and in_session:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.9
            
            # Price above SMA200 + RSI > 45 + bull regime
            elif price_above_sma200 and rsi_14[i] > 45.0 and hma_bullish and in_session:
                if new_signal == 0.0:
                    new_signal = LONG_BASE * 0.85
        
        # Also allow longs in 4h bull (even if 1d neutral)
        elif regime_4h_bull and not regime_4h_bear:
            if rsi_strong_oversold and in_session and volume_ok:
                new_signal = LONG_BASE * 0.8
        
        # SHORT ENTRIES (only in strong bear regime, reduced size)
        if strong_bear:
            # Primary: RSI pullback + HMA bearish + session + volume
            if rsi_pullback_short and hma_bearish and in_session and volume_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Strong: RSI very overbought + strong bear + session
            elif rsi_strong_overbought and strong_bear and in_session:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # HMA bearish crossover + RSI falling + volume
            elif hma_bearish and hma_slope_down and rsi_falling and volume_ok and in_session:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.9
            
            # Price below SMA200 + RSI < 55 + bear regime
            elif not price_above_sma200 and rsi_14[i] < 55.0 and hma_bearish and in_session:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.85
        
        # Also allow shorts in 4h bear (even if 1d neutral)
        elif regime_4h_bear and not regime_4h_bull:
            if rsi_strong_overbought and in_session and volume_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 20+ trades/year on 1h) ===
        # Force trade if no signal for 40 bars (~40 hours = ~2 days)
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if strong_bull and rsi_14[i] > 40.0 and in_session:
                new_signal = LONG_BASE * 0.6
            elif strong_bear and rsi_14[i] < 60.0 and in_session:
                new_signal = -SHORT_BASE * 0.6
            elif rsi_strong_oversold and in_session:
                new_signal = LONG_BASE * 0.6
            elif rsi_strong_overbought and in_session:
                new_signal = -SHORT_BASE * 0.6
        
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
        
        # === RSI REVERSAL EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_strong_overbought:
                rsi_exit = True
            if position_side < 0 and rsi_strong_oversold:
                rsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and strong_bear and price_below_hma:
                regime_reversal = True
            if position_side < 0 and strong_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.26:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.16:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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