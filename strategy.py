#!/usr/bin/env python3
"""
Experiment #288: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness + Session Filter

Hypothesis: After #278 failed on 30m (Sharpe=-2.112, too many trades), implement
VERY STRICT entry conditions with 3+ confluence filters to limit trades to 30-80/year.

Key components:
1. 1d HMA for PRIMARY trend direction (HTF filter)
2. 4h Choppiness Index for regime detection (trend vs mean-revert)
3. Connors RSI (CRSI) on 30m for precise entry timing
4. Session filter: only 8-20 UTC (high liquidity, less noise)
5. Volume filter: >0.8x 20-bar average
6. Asymmetric sizing: smaller positions for mean-revert, larger for trend

Position sizing: 0.20 base, 0.25 strong (conservative for 30m TF)
Target: 30-80 trades/year (strict confluence = fewer but higher quality)
Stoploss: 2.5 * ATR trailing

CRITICAL: Use mtf_data helper for HTF - call get_htf_data() ONCE before loop!
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_connors_chop_session_vol_4h1d_v1"
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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of prior closes lower than current close
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak_abs[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 - streak_abs[i] * 10)
    
    # Component 3: Percent Rank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_lower = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100 * count_lower / (rank_period - 1)
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.fillna(0).values

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
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
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Calculate 4h HTF indicators
    chop_4h_14 = calculate_choppiness_index(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        14
    )
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    chop_4h_14_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_14)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    hma_30m_21 = calculate_hma(close, 21)
    hma_30m_50 = calculate_hma(close, 50)
    
    # Volume moving average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract hours for session filter
    hours = np.array([extract_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, conservative for 30m)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.25
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_30m_21[i]):
            continue
        
        if np.isnan(chop_4h_14_aligned[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 1D TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_50_aligned[i]
        regime_bear = close[i] < hma_1d_50_aligned[i]
        
        # === 4H CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (trend follow entries)
        is_choppy = chop_4h_14_aligned[i] > 55.0
        is_trending = chop_4h_14_aligned[i] < 45.0
        
        # === TREND STRENGTH ===
        is_strong_trend = adx_14[i] > 25.0
        is_weak_trend = adx_14[i] < 20.0
        
        # === 30M LOCAL SIGNALS ===
        price_above_30m_hma = close[i] > hma_30m_21[i]
        price_below_30m_hma = close[i] < hma_30m_21[i]
        hma_30m_bullish = hma_30m_21[i] > hma_30m_50[i]
        hma_30m_bearish = hma_30m_21[i] < hma_30m_50[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_extreme_oversold = crsi[i] < 10.0
        crsi_extreme_overbought = crsi[i] > 90.0
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Only trade during session with volume
        if not (in_session and volume_ok):
            signals[i] = 0.0
            continue
        
        # TREND FOLLOWING MODE (when trending + strong ADX + regime aligned)
        if is_trending and is_strong_trend:
            # LONG: 4 confluence = Trending + bull regime + 30m HMA bullish + CRSI neutral-bull
            confluence_long = (
                regime_bull and 
                hma_30m_bullish and 
                price_above_30m_hma and
                crsi[i] > 40 and crsi[i] < 70
            )
            if confluence_long:
                new_signal = STRONG_SIZE
            
            # SHORT: 4 confluence = Trending + bear regime + 30m HMA bearish + CRSI neutral-bear
            confluence_short = (
                regime_bear and 
                hma_30m_bearish and 
                price_below_30m_hma and
                crsi[i] > 30 and crsi[i] < 60
            )
            if confluence_short and new_signal == 0.0:
                new_signal = -STRONG_SIZE
        
        # MEAN REVERSION MODE (when choppy + weak ADX)
        if is_choppy or is_weak_trend:
            # LONG: 3 confluence = Choppy + CRSI extreme oversold + not strong bear regime
            confluence_mr_long = (
                crsi_extreme_oversold and
                not regime_bear and
                volume_ok
            )
            if confluence_mr_long:
                new_signal = BASE_SIZE
            
            # SHORT: 3 confluence = Choppy + CRSI extreme overbought + not strong bull regime
            confluence_mr_short = (
                crsi_extreme_overbought and
                not regime_bull and
                volume_ok
            )
            if confluence_mr_short and new_signal == 0.0:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (ensure 10+ trades) ===
        # Force trade if no signal for 25 bars (~12.5 hours on 30m)
        if bars_since_last_trade > 25 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] > 35 and price_above_30m_hma and in_session:
                new_signal = BASE_SIZE * 0.8
            elif regime_bear and crsi[i] < 65 and price_below_30m_hma and in_session:
                new_signal = -BASE_SIZE * 0.8
        
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
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and hma_30m_bearish:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and hma_30m_bullish:
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