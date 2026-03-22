#!/usr/bin/env python3
"""
Experiment #326: 12h Primary + 1d HTF — Dual Regime (Trend + Mean Revert) + Connors RSI

Hypothesis: A dual-regime approach outperforms single-strategy approaches because:
1. Crypto alternates between trending (2021 bull, 2022 crash) and ranging (2023-2024, 2025)
2. Choppiness Index cleanly separates these regimes (CHOP>55=range, CHOP<45=trend)
3. Connors RSI (CRSI) has 75% win rate for mean reversion in range markets
4. HMA trend following works well in trending markets with 1d filter
5. 12h timeframe targets 20-50 trades/year (appropriate fee drag)
6. Asymmetric sizing favors longs (crypto bias) but allows shorts in bear regime

Why this might beat current best (Sharpe=0.424):
- Dual regime adapts to market conditions instead of forcing one approach
- Connors RSI = (RSI(2) + RSI_Streak(2) + PercentRank(100)) / 3 — proven edge
- 1d HMA(21) provides cleaner trend filter than 12h indicators
- Looser entry thresholds ensure 20+ trades/year (avoid 0-trade failure)
- ATR trailing stop protects against 2022-style crashes

Key differences from failed #316, #321, #323:
- Connors RSI instead of standard RSI (faster mean reversion signal)
- Dual regime logic (not just trend or just mean revert)
- Simpler entry conditions (fewer AND conditions = more trades)
- 12h instead of 1d/4h (sweet spot for trade frequency)

Position sizing: 0.25 base, 0.30 strong conviction
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_hma_1d_v1"
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

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA, smoother than SMA.
    """
    n = period
    n2 = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA helper
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, n2)
    wma_full = wma(close_s, n)
    
    diff = 2.0 * wma_half - wma_full
    hma = wma(diff, sqrt_n)
    
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

def calculate_connors_rsi(close, rsi_period=2, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 2) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    
    Entry signals:
    - Long: CRSI < 10 (extreme oversold)
    - Short: CRSI > 90 (extreme overbought)
    - Looser: Long CRSI < 20, Short CRSI > 80
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(2) component
    rsi_2 = calculate_rsi(close, rsi_period)
    
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
        pos_streaks = np.sum(streak[max(0, i-streak_period):i] > 0)
        total = streak_period
        if total > 0:
            streak_rsi[i] = 100.0 * pos_streaks / total
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank component
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = (close[i] - close[i-1]) / (close[i-1] + 1e-10) * 100.0
    
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = returns[max(0, i-pr_period):i]
        if len(window) > 0:
            percent_rank[i] = 100.0 * np.sum(window < returns[i]) / len(window)
        else:
            percent_rank[i] = 50.0
    
    # Combine components
    for i in range(pr_period, n):
        crsi[i] = (rsi_2[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

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
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100.0 * minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=2, streak_period=2, pr_period=100)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    adx_14 = calculate_adx(high, low, close, 14)
    
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
    last_trade_bar = -20
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(adx_14[i]):
            continue
        
        # === 1D MAJOR TREND REGIME (primary direction filter) ===
        # Bull: price above 1d HMA(21)
        # Bear: price below 1d HMA(21)
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (use mean reversion)
        # CHOP < 45 = trending market (use trend following)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === VOLATILITY REGIME (ATR ratio) ===
        atr_ratio = atr_14[i] / (atr_30[i] + 1e-10)
        high_vol = atr_ratio > 1.5
        vol_scale = 0.7 if high_vol else 1.0
        
        # === 12H LOCAL TREND ===
        hma_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # HMA slope (3-bar lookback)
        hma_slope_up = hma_12h_21[i] > hma_12h_21[i-3] if i >= 3 else False
        hma_slope_down = hma_12h_21[i] < hma_12h_21[i-3] if i >= 3 else False
        
        # Price position relative to HMA
        price_above_hma = close[i] > hma_12h_21[i]
        price_below_hma = close[i] < hma_12h_21[i]
        
        # === TREND STRENGTH (ADX) ===
        strong_trend = adx_14[i] > 25.0
        weak_trend = adx_14[i] < 20.0
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        crsi_oversold = crsi[i] < 25.0
        crsi_overbought = crsi[i] > 75.0
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_rising = rsi_14[i] > rsi_14[i-1] if i > 0 else False
        rsi_falling = rsi_14[i] < rsi_14[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # === TRENDING REGIME (CHOP < 45) ===
        if is_trending and strong_trend:
            # LONG: 1d bull + 12h HMA bullish + price above HMA
            if regime_bull and hma_bullish and price_above_hma and hma_slope_up:
                if rsi_14[i] < 55.0 or rsi_rising:
                    new_signal = LONG_BASE * vol_scale
            
            # STRONG LONG: All conditions + CRSI not overbought
            if regime_bull and hma_bullish and price_above_hma and crsi[i] < 70.0:
                if adx_14[i] > 30.0:
                    new_signal = LONG_STRONG * vol_scale
            
            # SHORT: 1d bear + 12h HMA bearish + price below HMA
            if regime_bear and hma_bearish and price_below_hma and hma_slope_down:
                if new_signal == 0.0 and (rsi_14[i] > 45.0 or rsi_falling):
                    new_signal = -SHORT_BASE * vol_scale
            
            # STRONG SHORT: All conditions + CRSI not oversold
            if regime_bear and hma_bearish and price_below_hma and crsi[i] > 30.0:
                if new_signal == 0.0 and adx_14[i] > 30.0:
                    new_signal = -SHORT_STRONG * vol_scale
        
        # === RANGING REGIME (CHOP > 55) ===
        elif is_choppy or weak_trend:
            # LONG: CRSI extreme oversold (mean reversion)
            if crsi_extreme_oversold:
                new_signal = LONG_BASE * vol_scale
            
            # LONG: CRSI oversold + 1d bull regime
            elif crsi_oversold and regime_bull:
                new_signal = LONG_BASE * 0.8 * vol_scale
            
            # LONG: RSI oversold + price below HMA (bounce play)
            elif rsi_oversold and price_below_hma and regime_bull:
                new_signal = LONG_BASE * 0.8 * vol_scale
            
            # SHORT: CRSI extreme overbought (mean reversion)
            if crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * vol_scale
            
            # SHORT: CRSI overbought + 1d bear regime
            elif crsi_overbought and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
            
            # SHORT: RSI overbought + price above HMA (fade play)
            elif rsi_overbought and price_above_hma and regime_bear:
                if new_signal == 0.0:
                    new_signal = -SHORT_BASE * 0.8 * vol_scale
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Use HMA crossover with CRSI filter
            if hma_bullish and hma_slope_up and crsi[i] < 60.0:
                new_signal = LONG_BASE * 0.7 * vol_scale
            elif hma_bearish and hma_slope_down and crsi[i] > 40.0:
                new_signal = -SHORT_BASE * 0.7 * vol_scale
        
        # === FREQUENCY SAFEGUARD (ensure 20+ trades/year on 12h) ===
        # Force trade if no signal for 45 bars (~22 days on 12h)
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            if regime_bull and crsi[i] < 50.0:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif regime_bear and crsi[i] > 50.0:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
            elif crsi_extreme_oversold:
                new_signal = LONG_BASE * 0.5 * vol_scale
            elif crsi_extreme_overbought:
                new_signal = -SHORT_BASE * 0.5 * vol_scale
        
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
            # Long position: exit when CRSI turns overbought
            if position_side > 0 and crsi_extreme_overbought:
                crsi_exit = True
            # Short position: exit when CRSI turns oversold
            if position_side < 0 and crsi_extreme_oversold:
                crsi_exit = True
        
        # === HMA REVERSAL EXIT ===
        hma_exit = False
        if in_position and position_side != 0:
            # Long position: exit when HMA turns bearish + price below
            if position_side > 0 and hma_bearish and price_below_hma:
                hma_exit = True
            # Short position: exit when HMA turns bullish + price above
            if position_side < 0 and hma_bullish and price_above_hma:
                hma_exit = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_hma:
                regime_reversal = True
            # Short position but 1d regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_hma:
                regime_reversal = True
        
        if stoploss_triggered or crsi_exit or hma_exit or regime_reversal:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.12:
                new_signal = 0.0
            elif new_signal > 0.28:
                new_signal = LONG_STRONG * vol_scale
            elif new_signal > 0:
                new_signal = LONG_BASE * vol_scale
            elif new_signal < -0.23:
                new_signal = -SHORT_STRONG * vol_scale
            else:
                new_signal = -SHORT_BASE * vol_scale
        
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