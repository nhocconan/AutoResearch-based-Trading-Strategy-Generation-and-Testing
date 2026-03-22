#!/usr/bin/env python3
"""
Experiment #292: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + Dual HMA Regime + RSI Pullback

Hypothesis: Building on #282 (Sharpe=0.351), improve with:
1. KAMA(14) on 12h for adaptive trend (responds to volatility, worked in #291 with +86% return)
2. Dual HTF regime: 1w HMA(21) for PRIMARY direction, 1d HMA(21) for confirmation
3. Choppiness(14) + ADX(14) for regime detection (trend vs mean-revert mode)
4. RSI(14) pullback entries in trend mode, RSI extremes in chop mode
5. Donchian(20) breakout for trend confirmation
6. Tighter stoploss: 2.0*ATR (vs 2.5) for better risk control
7. Regime-adaptive sizing: 0.30 strong trend, 0.20 choppy/uncertain

Key improvements over #282:
- KAMA instead of HMA on primary (adaptive to volatility regimes)
- Add 1w HTF for long-term regime filter (better trend alignment)
- Tighter stoploss reduces drawdown
- Discrete sizing reduces fee churn

Position sizing: 0.20-0.30 discrete levels
Target: 20-50 trades/year per symbol (appropriate for 12h)
Stoploss: 2.0 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_dual_hma_rsi_chop_1d1w_v1"
timeframe = "12h"
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

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market noise: fast in trends, slow in chop.
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    n = period
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    close_s = pd.Series(close)
    
    # Efficiency Ratio
    price_change = np.abs(close_s.diff(n))
    vol_sum = np.abs(close_s.diff()).rolling(window=n, min_periods=n).sum()
    er = price_change / vol_sum.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (long-term regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d HTF indicators (medium-term trend)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_12h_14 = calculate_kama(close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    STRONG_SIZE = 0.30  # Strong trend conviction
    BASE_SIZE = 0.20    # Base/uncertain regime
    CHOP_SIZE = 0.15    # Mean reversion in chop
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_12h_14[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1W LONG-TERM REGIME (primary direction filter) ===
        # Bull: price above 1w HMA
        # Bear: price below 1w HMA
        regime_bull_1w = close[i] > hma_1w_21_aligned[i]
        regime_bear_1w = close[i] < hma_1w_21_aligned[i]
        
        # === 1D MEDIUM-TERM TREND (confirmation) ===
        regime_bull_1d = close[i] > hma_1d_21_aligned[i]
        regime_bear_1d = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = range market (mean revert entries)
        # CHOP < 38.2 = trending market (breakout entries)
        is_choppy = chop_14[i] > 58.0
        is_trending = chop_14[i] < 42.0
        
        # === TREND STRENGTH ===
        is_strong_trend = adx_14[i] > 25.0
        is_weak_trend = adx_14[i] < 20.0
        
        # === 12H LOCAL SIGNALS ===
        price_above_kama = close[i] > kama_12h_14[i]
        price_below_kama = close[i] < kama_12h_14[i]
        kama_slope_up = kama_12h_14[i] > kama_12h_14[i-5] if i >= 5 else False
        kama_slope_down = kama_12h_14[i] < kama_12h_14[i-5] if i >= 5 else False
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.999
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.001
        
        # === RSI THRESHOLDS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        rsi_neutral_bull = 45.0 < rsi_14[i] < 60.0
        rsi_neutral_bear = 40.0 < rsi_14[i] < 55.0
        
        # === REGIME CONFLUENCE ===
        # Strong bull: 1w bull + 1d bull + price above KAMA
        strong_bull = regime_bull_1w and regime_bull_1d and price_above_kama
        # Strong bear: 1w bear + 1d bear + price below KAMA
        strong_bear = regime_bear_1w and regime_bear_1d and price_below_kama
        # Aligned bull: at least 2 of 3 bullish
        aligned_bull = (regime_bull_1w + regime_bull_1d + price_above_kama) >= 2
        # Aligned bear: at least 2 of 3 bearish
        aligned_bear = (regime_bear_1w + regime_bear_1d + price_below_kama) >= 2
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + strong ADX + regime aligned)
        if is_trending and is_strong_trend:
            # LONG: Strong bull regime + RSI confirming (not overbought)
            if strong_bull and rsi_14[i] < 70:
                new_signal = STRONG_SIZE
            # LONG: Aligned bull + Donchian breakout + KAMA slope up
            elif aligned_bull and donchian_breakout_long and kama_slope_up:
                new_signal = BASE_SIZE
            
            # SHORT: Strong bear regime + RSI confirming (not oversold)
            if strong_bear and rsi_14[i] > 30:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Aligned bear + Donchian breakdown + KAMA slope down
            elif aligned_bear and donchian_breakout_short and kama_slope_down:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy + weak ADX)
        if is_choppy or is_weak_trend:
            # LONG: Choppy + RSI oversold (<35) + not in strong bear regime
            if rsi_oversold and not strong_bear:
                new_signal = CHOP_SIZE
            # LONG: Choppy + RSI extreme oversold (<25) in any regime
            if rsi_extreme_oversold:
                if new_signal == 0.0 or abs(new_signal) < CHOP_SIZE:
                    new_signal = CHOP_SIZE
            
            # SHORT: Choppy + RSI overbought (>65) + not in strong bull regime
            if rsi_overbought and not strong_bull:
                if new_signal == 0.0:
                    new_signal = -CHOP_SIZE
            # SHORT: Choppy + RSI extreme overbought (>75) in any regime
            if rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -CHOP_SIZE
        
        # === PULLBACK ENTRY IN TREND (high probability setup) ===
        # Long pullback: Strong bull regime + RSI pulled back to 40-50 + price > KAMA
        if strong_bull and 40.0 < rsi_14[i] < 52.0 and price_above_kama:
            if new_signal == 0.0:
                new_signal = BASE_SIZE
        
        # Short pullback: Strong bear regime + RSI rallied to 48-60 + price < KAMA
        if strong_bear and 48.0 < rsi_14[i] < 60.0 and price_below_kama:
            if new_signal == 0.0:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (ensure 10+ trades) ===
        # Force trade if no signal for 15 bars (~180h = 7.5 days on 12h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if strong_bull and rsi_14[i] > 40:
                new_signal = BASE_SIZE * 0.8
            elif strong_bear and rsi_14[i] < 60:
                new_signal = -BASE_SIZE * 0.8
            elif is_choppy and rsi_14[i] < 35:
                new_signal = CHOP_SIZE * 0.8
            elif is_choppy and rsi_14[i] > 65:
                new_signal = -CHOP_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish (both 1w and 1d)
            if position_side > 0 and regime_bear_1w and regime_bear_1d:
                regime_reversal = True
            # Short position but regime turns strongly bullish (both 1w and 1d)
            if position_side < 0 and regime_bull_1w and regime_bull_1d:
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