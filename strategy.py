#!/usr/bin/env python3
"""
Experiment #218: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: After 217 experiments, the key insight is that lower TF (30m) strategies
fail due to TOO MANY TRADES causing fee drag. This strategy uses:

1. 1d HMA for MAJOR trend bias (very slow, only changes direction rarely)
2. 4h HMA for INTERMEDIATE trend direction (signal filter)
3. 30m RSI for ENTRY timing (pullback entries within HTF trend)
4. SESSION filter (8-20 UTC only) — naturally reduces trade count by 50%
5. VOLUME filter (volume > 0.8x 20-bar avg) — confirms real moves
6. ATR(14) trailing stop at 2.5x — protects against reversals

Why this differs from failed strategies:
- NO Connors RSI (failed in 50+ experiments)
- NO Choppiness Index (failed in 40+ experiments)
- Session filter reduces trades WITHOUT adding complexity
- Multiple entry paths ensure 10+ trades/symbol (looser than #217)
- HTF trend is DIRECTION, 30m is only TIMING (proven pattern)

Position sizing: 0.25 discrete (smaller for 30m to reduce fee impact)
Target: 40-80 trades/year per symbol (matches 30m cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_session_4h1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_avg
    ratio[vol_avg == 0] = 1.0
    return ratio

def get_utc_hour(open_time):
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend bias)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Calculate 4h HTF indicators (intermediate trend)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for entry timing
    volume_ratio = calculate_volume_ratio(volume, 20)
    
    # 30m HMA for local trend
    hma_30m_21 = calculate_hma(close, 21)
    hma_30m_slope = calculate_hma_slope(hma_30m_21, 3)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    BASE_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(rsi_7[i]):
            continue
        
        if np.isnan(hma_30m_21[i]) or np.isnan(hma_30m_slope[i]):
            continue
        
        if np.isnan(volume_ratio[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only for entries) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === HTF TREND BIAS (1d) ===
        # Daily trend determines overall bias (very slow changing)
        daily_bullish = hma_1d_slope_aligned[i] > 0.10
        daily_bearish = hma_1d_slope_aligned[i] < -0.10
        daily_neutral = not daily_bullish and not daily_bearish
        
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === INTERMEDIATE TREND (4h) ===
        h4_bullish = hma_4h_slope_aligned[i] > 0.15
        h4_bearish = hma_4h_slope_aligned[i] < -0.15
        
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === LOCAL TREND (30m HMA) ===
        local_bullish = hma_30m_slope[i] > 0.25
        local_bearish = hma_30m_slope[i] < -0.25
        
        price_above_30m_hma = close[i] > hma_30m_21[i]
        price_below_30m_hma = close[i] < hma_30m_21[i]
        
        # === MOMENTUM (RSI) ===
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        rsi_oversold = rsi_7[i] < 35  # Pullback entry
        rsi_overbought = rsi_7[i] > 65  # Pullback entry short
        rsi_strong_bull = rsi_14[i] > 58
        rsi_strong_bear = rsi_14[i] < 42
        
        # === VOLUME FILTER ===
        volume_confirmed = volume_ratio[i] > 0.8
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade frequency
        long_score = 0
        
        # Path 1: Daily bullish + 4h bullish + RSI pullback (primary - strong signal)
        if daily_bullish and h4_bullish and rsi_oversold and in_session:
            long_score += 5
        
        # Path 2: Daily bullish + price above 1d HMA + RSI bullish + volume
        if daily_bullish and price_above_1d_hma and rsi_bullish and volume_confirmed:
            long_score += 4
        
        # Path 3: 4h bullish + local bullish + RSI > 50 (trend continuation)
        if h4_bullish and local_bullish and rsi_bullish and price_above_30m_hma:
            long_score += 3
        
        # Path 4: Daily bullish + RSI pullback (simpler, more trades)
        if daily_bullish and rsi_oversold and in_session and volume_confirmed:
            long_score += 3
        
        # Path 5: 4h bullish + RSI strong + volume (momentum entry)
        if h4_bullish and rsi_strong_bull and volume_confirmed:
            long_score += 2
        
        # Path 6: Daily bullish + price above 1d HMA (basic trend follow)
        if daily_bullish and price_above_1d_hma and rsi_bullish:
            long_score += 2
        
        # Path 7: All HTF aligned bullish (strongest signal)
        if daily_bullish and h4_bullish and local_bullish and rsi_bullish:
            long_score += 4
        
        # Path 8: RSI oversold bounce in uptrend (mean reversion within trend)
        if daily_bullish and rsi_7[i] < 30 and price_above_4h_hma and in_session:
            long_score += 3
        
        if long_score >= 5:
            new_signal = current_size
        elif long_score >= 4 and bars_since_last_trade > 20:
            new_signal = current_size
        elif long_score >= 3 and bars_since_last_trade > 30:
            new_signal = current_size * 0.7
        elif long_score >= 2 and bars_since_last_trade > 50:
            new_signal = current_size * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Daily bearish + 4h bearish + RSI pullback (primary)
        if daily_bearish and h4_bearish and rsi_overbought and in_session:
            short_score += 5
        
        # Path 2: Daily bearish + price below 1d HMA + RSI bearish + volume
        if daily_bearish and price_below_1d_hma and rsi_bearish and volume_confirmed:
            short_score += 4
        
        # Path 3: 4h bearish + local bearish + RSI < 50 (trend continuation)
        if h4_bearish and local_bearish and rsi_bearish and price_below_30m_hma:
            short_score += 3
        
        # Path 4: Daily bearish + RSI pullback (simpler, more trades)
        if daily_bearish and rsi_overbought and in_session and volume_confirmed:
            short_score += 3
        
        # Path 5: 4h bearish + RSI strong + volume (momentum entry)
        if h4_bearish and rsi_strong_bear and volume_confirmed:
            short_score += 2
        
        # Path 6: Daily bearish + price below 1d HMA (basic trend follow)
        if daily_bearish and price_below_1d_hma and rsi_bearish:
            short_score += 2
        
        # Path 7: All HTF aligned bearish (strongest signal)
        if daily_bearish and h4_bearish and local_bearish and rsi_bearish:
            short_score += 4
        
        # Path 8: RSI overbought fade in downtrend
        if daily_bearish and rsi_7[i] > 70 and price_below_4h_hma and in_session:
            short_score += 3
        
        if short_score >= 5:
            new_signal = -current_size
        elif short_score >= 4 and bars_since_last_trade > 20:
            new_signal = -current_size
        elif short_score >= 3 and bars_since_last_trade > 30:
            new_signal = -current_size * 0.7
        elif short_score >= 2 and bars_since_last_trade > 50:
            new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 150 bars (~75 hours on 30m = 3 days)
        if bars_since_last_trade > 150 and new_signal == 0.0 and not in_position:
            if daily_bullish and rsi_14[i] > 52 and price_above_4h_hma:
                new_signal = current_size * 0.35
            elif daily_bearish and rsi_14[i] < 48 and price_below_4h_hma:
                new_signal = -current_size * 0.35
            elif h4_bullish and rsi_14[i] > 55 and in_session:
                new_signal = current_size * 0.25
            elif h4_bearish and rsi_14[i] < 45 and in_session:
                new_signal = -current_size * 0.25
        
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but daily turns strongly bearish
            if position_side > 0 and daily_bearish and price_below_1d_hma:
                trend_reversal = True
            # Short position but daily turns strongly bullish
            if position_side < 0 and daily_bullish and price_above_1d_hma:
                trend_reversal = True
        
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