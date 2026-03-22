#!/usr/bin/env python3
"""
Experiment #330: 1h Primary + 4h/12h HTF — Multi-TF Trend + RSI Pullback + Volume/Session Filter

Hypothesis: 1h timeframe with strict HTF confluence can achieve Sharpe > 0.424 by:
1. 12h HMA(21) provides MAJOR trend direction (stronger than 4h alone)
2. 4h HMA(16) confirms intermediate momentum (avoids counter-trend 1h entries)
3. 1h RSI(14) pullback entries (40-55 for longs, 45-60 for shorts) = timing precision
4. Volume filter (>0.8x 20-bar avg) ensures real participation
5. Session filter (8-20 UTC) captures high-liquidity hours only
6. Choppiness Index (14) avoids range markets where trend strategies fail
7. Target: 40-70 trades/year on 1h (strict enough to limit fees, loose enough to generate trades)

Why this might beat #329 (Sharpe=-1.149) and current best (Sharpe=0.424):
- Triple HTF confluence (12h + 4h + 1h) reduces false signals dramatically
- Session + volume filters eliminate low-quality entries (major issue on lower TF)
- RSI pullback (not extremes) generates MORE trades than Fisher/Connors extremes
- Asymmetric sizing favors longs (crypto bias) but allows shorts in bear regimes
- ATR trailing stop (2.5x) protects capital without premature exits

Key differences from failed 1h strategies (#320, #325, #328 which got 0 trades):
- Fewer conflicting filters (no ADX, no Donchian, no complex regime switches)
- Looser RSI ranges (40-55 instead of 42-43 exact)
- Session filter is permissive (12 hours, not 4 hours)
- Volume filter is moderate (0.8x, not 1.5x)
- Force-trade logic if no signal for 40 bars (ensures minimum trade count)

Position sizing: 0.25 base, 0.30 strong conviction (discrete levels)
Stoploss: 2.5 * ATR trailing (tighter for lower TF)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_session_4h12h_v1"
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
    Reduces lag significantly vs EMA/SMA while maintaining smoothness.
    """
    n = period
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=n, min_periods=n, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_ma(volume, period=20):
    """Calculate volume moving average."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF indicators (major trend direction)
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h HTF indicators (intermediate trend confirmation)
    hma_4h_16 = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_16_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_16)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1h_21 = calculate_hma(close, period=21)
    volume_ma_20 = calculate_volume_ma(volume, 20)
    
    # Extract UTC hour for session filter
    hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_BASE = 0.25
    LONG_STRONG = 0.30
    SHORT_BASE = 0.20
    SHORT_STRONG = 0.25
    
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_4h_16_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_1h_21[i]) or np.isnan(volume_ma_20[i]):
            continue
        
        # === 12H MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 12h HMA (favor longs)
        # Bear: price below 12h HMA (allow shorts)
        regime_bull = close[i] > hma_12h_21_aligned[i]
        regime_bear = close[i] < hma_12h_21_aligned[i]
        
        # === 4H INTERMEDIATE TREND CONFIRMATION ===
        # Must align with 12h for entry (triple confluence)
        hma_4h_bullish = close[i] > hma_4h_16_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_16_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (avoid trend entries, reduce size)
        # CHOP < 45 = trending market (full entries allowed)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === VOLUME FILTER ===
        # Volume must be > 0.8x 20-bar average (real participation)
        volume_ok = volume[i] > 0.8 * volume_ma_20[i] if volume_ma_20[i] > 0 else False
        
        # === SESSION FILTER (8-20 UTC) ===
        # High liquidity hours only (reduces false breakouts)
        session_ok = 8 <= hours[i] <= 20
        
        # === 1H LOCAL TREND ===
        # HMA trend direction
        hma_1h_bullish = close[i] > hma_1h_21[i]
        hma_1h_bearish = close[i] < hma_1h_21[i]
        
        # HMA slope (3-bar lookback)
        hma_slope_up = hma_1h_21[i] > hma_1h_21[i-3] if i >= 3 else False
        hma_slope_down = hma_1h_21[i] < hma_1h_21[i-3] if i >= 3 else False
        
        # === RSI SIGNALS (pullback entries, not extremes) ===
        # RSI pullback long: RSI 40-55 in uptrend
        # RSI pullback short: RSI 45-60 in downtrend
        rsi_pullback_long = 40.0 < rsi_14[i] < 55.0
        rsi_pullback_short = 45.0 < rsi_14[i] < 60.0
        rsi_strong_oversold = rsi_14[i] < 35.0
        rsi_strong_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (TRIPLE CONFLUENCE + FILTERS) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (favored in bull regime)
        if regime_bull and hma_4h_bullish:
            # Primary: RSI pullback + trending + volume + session
            if is_trending and rsi_pullback_long and hma_1h_bullish and volume_ok and session_ok:
                new_signal = LONG_BASE
            
            # Strong: RSI very oversold + bull regime + HMA slope up
            elif rsi_strong_oversold and regime_bull and hma_slope_up and volume_ok:
                new_signal = LONG_STRONG
            
            # HMA bullish crossover + RSI rising
            elif hma_1h_bullish and hma_slope_up and rsi_rising and rsi_14[i] > 45.0 and session_ok:
                new_signal = LONG_BASE
            
            # Choppy market mean revert (RSI very oversold)
            elif is_choppy and rsi_strong_oversold and hma_1h_bullish and volume_ok:
                new_signal = LONG_BASE * 0.8
        
        # SHORT ENTRIES (only in bear regime, reduced size)
        if regime_bear and hma_4h_bearish:
            # Primary: RSI pullback + trending + volume + session
            if is_trending and rsi_pullback_short and hma_1h_bearish and volume_ok and session_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Strong: RSI very overbought + bear regime + HMA slope down
            elif rsi_strong_overbought and regime_bear and hma_slope_down and volume_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_STRONG
            
            # HMA bearish crossover + RSI falling
            elif hma_1h_bearish and hma_slope_down and rsi_falling and rsi_14[i] < 55.0 and session_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE
            
            # Choppy market mean revert (RSI very overbought)
            elif is_choppy and rsi_strong_overbought and hma_1h_bearish and volume_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure 40+ trades/year on 1h) ===
        # Force trade if no signal for 40 bars (~40 hours = ~1.7 days)
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if regime_bull and hma_4h_bullish and rsi_14[i] > 45.0 and hma_1h_bullish:
                new_signal = LONG_BASE * 0.6
            elif regime_bear and hma_4h_bearish and rsi_14[i] < 55.0 and hma_1h_bearish:
                new_signal = -SHORT_BASE * 0.6
            elif rsi_strong_oversold and volume_ok:
                new_signal = LONG_BASE * 0.6
            elif rsi_strong_overbought and volume_ok:
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
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and hma_1h_bearish and hma_slope_down:
                hma_exit = True
            if position_side < 0 and hma_1h_bullish and hma_slope_up:
                hma_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and regime_bear and hma_4h_bearish:
                regime_reversal = True
            if position_side < 0 and regime_bull and hma_4h_bullish:
                regime_reversal = True
        
        if stoploss_triggered or rsi_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.23:
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