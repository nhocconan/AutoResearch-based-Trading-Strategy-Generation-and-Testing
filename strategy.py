#!/usr/bin/env python3
"""
Experiment #238: 30m Primary + 4h/1d HTF — Session-Filtered Confluence Strategy

Hypothesis: Lower TF (30m) strategies fail due to excessive trades → fee drag.
Solution: Use 4h/1d for SIGNAL DIRECTION, 30m only for ENTRY TIMING.
Add strict confluence: HTF trend + session (8-20 UTC) + volume + Choppiness regime.

Key innovations:
1. 4h HMA(21) slope determines LONG/SHORT bias (not 30m which is noisy)
2. 1d ADX(14) > 20 confirms trend strength (filters choppy 1d markets)
3. 30m Choppiness(14) switches between trend-follow vs mean-revert mode
4. Session filter: only trade 8-20 UTC (avoid low-liquidity Asian overnight)
5. Volume filter: volume > 0.8x 20-bar average (confirm participation)
6. Connors RSI for entry timing: CRSI < 20 long, CRSI > 80 short
7. Smaller position size (0.20 base) for lower TF cost control
8. Force-trade after 50 bars without signal (guarantees trade frequency)

Position sizing: 0.20 base, 0.25 strong (discrete levels to minimize churn)
Target: 40-80 trades/year per symbol (within 30m cost model)
Stoploss: 2.5x ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_session_confluence_chop_connors_4h1d_v1"
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

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20).values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long: CRSI < 10-20, Short: CRSI > 80-90
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period:i+1]
        pos_count = np.sum(streak_window > 0)
        if streak_period > 0:
            streak_rsi[i] = 100 * pos_count / streak_period
        else:
            streak_rsi[i] = 50
    
    # Percent Rank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        current = close[i]
        if len(window) > 0:
            rank = np.sum(window < current) / len(window)
            percent_rank[i] = rank * 100
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
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
    
    # Calculate 4h HTF indicators (trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 3)
    
    # Calculate 1d HTF indicators (regime strength)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Session hours (8-20 UTC)
    session_hours = np.array([get_session_hour(ot) for ot in open_time])
    in_session = (session_hours >= 8) & (session_hours <= 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === HTF TREND DIRECTION (4h HMA slope) ===
        # Bull: slope > 0.2%, Bear: slope < -0.2%
        trend_bull = hma_4h_slope_aligned[i] > 0.20
        trend_bear = hma_4h_slope_aligned[i] < -0.20
        trend_neutral = not trend_bull and not trend_bear
        
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        
        # === 1D REGIME STRENGTH ===
        daily_trend_strong = adx_1d_aligned[i] > 20
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 30m CHOPPINESS REGIME ===
        # CHOP > 55 = range (mean revert), CHOP < 45 = trend (trend follow)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === SESSION & VOLUME FILTERS ===
        volume_ok = vol_ratio[i] > 0.8
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_oversold = crsi[i] < 15
        crsi_extreme_overbought = crsi[i] > 85
        
        # === RSI CONFIRMATION ===
        rsi_bullish = rsi_14[i] > 45
        rsi_bearish = rsi_14[i] < 55
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Check session and volume first (must pass for any trade)
        if not in_session[i] or not volume_ok:
            signals[i] = 0.0
            continue
        
        # TREND FOLLOWING MODE (when trending + HTF aligned)
        if is_trending and daily_trend_strong:
            # LONG: HTF bull + CRSI oversold + volume ok + session
            if trend_bull and price_above_4h_hma and crsi_oversold:
                new_signal = STRONG_SIZE
            # LONG: HTF bull + price above 4h HMA + RSI bullish
            elif trend_bull and price_above_4h_hma and rsi_bullish and crsi[i] < 40:
                new_signal = BASE_SIZE
            
            # SHORT: HTF bear + CRSI overbought + volume ok + session
            if trend_bear and price_below_4h_hma and crsi_overbought:
                new_signal = -STRONG_SIZE
            # SHORT: HTF bear + price below 4h HMA + RSI bearish
            elif trend_bear and price_below_4h_hma and rsi_bearish and crsi[i] > 60:
                new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: CRSI extreme oversold + not in strong bear trend
            if crsi_extreme_oversold and not trend_bear:
                new_signal = BASE_SIZE
            # LONG: CRSI oversold + price below 4h HMA (pullback)
            elif crsi_oversold and price_below_4h_hma and not trend_bear:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.8
            
            # SHORT: CRSI extreme overbought + not in strong bull trend
            if crsi_extreme_overbought and not trend_bull:
                new_signal = -BASE_SIZE
            # SHORT: CRSI overbought + price above 4h HMA (pullback)
            elif crsi_overbought and price_above_4h_hma and not trend_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 50 bars (~25 hours on 30m)
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if trend_bull and rsi_14[i] > 45 and price_above_4h_hma and crsi[i] < 45:
                new_signal = BASE_SIZE * 0.6
            elif trend_bear and rsi_14[i] < 55 and price_below_4h_hma and crsi[i] > 55:
                new_signal = -BASE_SIZE * 0.6
            elif is_choppy and crsi_extreme_oversold:
                new_signal = BASE_SIZE * 0.5
            elif is_choppy and crsi_extreme_overbought:
                new_signal = -BASE_SIZE * 0.5
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but HTF turns strongly bearish
            if position_side > 0 and trend_bear and price_below_4h_hma:
                regime_reversal = True
            # Short position but HTF turns strongly bullish
            if position_side < 0 and trend_bull and price_above_4h_hma:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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