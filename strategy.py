#!/usr/bin/env python3
"""
Experiment #298: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness + Session Filter

Hypothesis: After 270+ failed experiments, focus on what ACTUALLY works for lower TF:
1. 4h HMA(21) for PRIMARY trend direction (not 1w - too slow for 30m entries)
2. 1d HMA(50) for SECONDARY trend confirmation (regime filter)
3. Connors RSI(3,2,100) for entry timing - proven 75% win rate on mean reversion
4. Choppiness Index(14) regime: CHOP>55=range(mean revert), CHOP<45=trend(follow)
5. Session filter: ONLY trade 8-20 UTC (highest liquidity, lowest noise)
6. Volume filter: volume > 0.7x 20-bar avg (confirm moves)
7. ATR(14) stoploss: 2.5x ATR trailing (tighter for 30m vs daily)

Why this might work when #288 failed (0 trades):
- #288 had TOO STRICT conditions (all filters must align perfectly)
- This uses SCORING system: need 3/4 confluence, not 4/4
- Relaxed RSI thresholds (CRSI<25 instead of <15) to ensure trades
- Session filter reduces noise but doesn't block all entries
- Target: 40-70 trades/year per symbol (appropriate for 30m)

Position sizing: 0.20 base, 0.30 strong conviction (smaller for lower TF)
Stoploss: 2.5 * ATR trailing (tighter than daily strategies)
Target trades: 40-70/year (balances fee drag vs opportunity)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_connors_chop_session_4h1d_v2"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    Long when CRSI < 10-25, Short when CRSI > 75-90
    Proven 75% win rate on mean reversion.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI(2) component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # PercentRank(100) component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period+1):i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (len(window) - 1) if len(window) > 1 else 0.5
        percent_rank[i] = rank * 100
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds)."""
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
    
    # Calculate 4h HTF indicators (primary trend)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    
    # Calculate 1d HTF indicators (secondary trend regime)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume average (20 bars)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours
    session_hours = np.array([calculate_session_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 30m)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    MIN_SIZE = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    bars_in_trade = 0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 4H TREND REGIME (primary direction filter) ===
        # Bull: price above 4h HMA(21) AND HMA(21) > HMA(50)
        # Bear: price below 4h HMA(21) AND HMA(21) < HMA(50)
        price_above_4h_hma = close[i] > hma_4h_21_aligned[i]
        price_below_4h_hma = close[i] < hma_4h_21_aligned[i]
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === 1D TREND REGIME (secondary confirmation) ===
        price_above_1d_hma = close[i] > hma_1d_50_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_50_aligned[i]
        
        # Strong bull: 4h bull + 1d bull
        # Strong bear: 4h bear + 1d bear
        strong_bull = price_above_4h_hma and hma_4h_bullish and price_above_1d_hma
        strong_bear = price_below_4h_hma and hma_4h_bearish and price_below_1d_hma
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= session_hours[i] <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / (vol_avg[i] + 1e-10)
        vol_confirmed = vol_ratio > 0.7
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        
        # === ENTRY SCORING SYSTEM (need 3/4 confluence) ===
        long_score = 0
        short_score = 0
        
        # Long scoring
        if strong_bull:
            long_score += 2
        elif price_above_4h_hma and hma_4h_bullish:
            long_score += 1
        
        if crsi_oversold:
            long_score += 2
        elif crsi[i] < 35:
            long_score += 1
        
        if is_choppy or is_trending:
            long_score += 1
        
        if in_session:
            long_score += 1
        
        if vol_confirmed:
            long_score += 1
        
        # Short scoring
        if strong_bear:
            short_score += 2
        elif price_below_4h_hma and hma_4h_bearish:
            short_score += 1
        
        if crsi_overbought:
            short_score += 2
        elif crsi[i] > 65:
            short_score += 1
        
        if is_choppy or is_trending:
            short_score += 1
        
        if in_session:
            short_score += 1
        
        if vol_confirmed:
            short_score += 1
        
        # === ENTRY LOGIC (relaxed for trade generation) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: Need score >= 5 (out of ~8 possible)
        if long_score >= 5:
            if crsi_extreme_oversold or (crsi_oversold and strong_bull):
                new_signal = STRONG_SIZE
            elif crsi_oversold and (in_session or vol_confirmed):
                new_signal = BASE_SIZE
        
        # SHORT: Need score >= 5 (out of ~8 possible)
        if short_score >= 5:
            if new_signal == 0.0:
                if crsi_extreme_overbought or (crsi_overbought and strong_bear):
                    new_signal = -STRONG_SIZE
                elif crsi_overbought and (in_session or vol_confirmed):
                    new_signal = -BASE_SIZE
        
        # === TRADE FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 40 bars (~20 hours on 30m)
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if strong_bull and crsi[i] < 40:
                new_signal = MIN_SIZE
            elif strong_bear and crsi[i] > 60:
                new_signal = -MIN_SIZE
            elif is_choppy and crsi_extreme_oversold:
                new_signal = MIN_SIZE
            elif is_choppy and crsi_extreme_overbought:
                new_signal = -MIN_SIZE
        
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
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI goes overbought
            if position_side > 0 and crsi[i] > 80:
                crsi_exit = True
            # Short position: exit when CRSI goes oversold
            if position_side < 0 and crsi[i] < 20:
                crsi_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 4h regime turns strongly bearish
            if position_side > 0 and strong_bear:
                regime_reversal = True
            # Short position but 4h regime turns strongly bullish
            if position_side < 0 and strong_bull:
                regime_reversal = True
        
        # Time-based exit (max 100 bars in trade ~50 hours)
        time_exit = False
        if in_position:
            bars_in_trade += 1
            if bars_in_trade > 100:
                time_exit = True
        else:
            bars_in_trade = 0
        
        if stoploss_triggered or crsi_exit or regime_reversal or time_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.15:
                new_signal = 0.0
            elif new_signal > 0.25:
                new_signal = STRONG_SIZE
            elif new_signal > 0:
                new_signal = BASE_SIZE
            elif new_signal < -0.25:
                new_signal = -STRONG_SIZE
            else:
                new_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                bars_in_trade = 0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
                bars_in_trade = 0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
                bars_in_trade = 0
        
        signals[i] = new_signal
    
    return signals