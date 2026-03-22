#!/usr/bin/env python3
"""
Experiment #243: 1h Multi-Regime Strategy with Funding Proxy + Volume Confirmation

Hypothesis: 1h timeframe captures intraday swings while 4h/12h HTF filters reduce noise.
Key insight from research: funding rate mean reversion has Sharpe 0.8-1.5 for BTC/ETH.
Since funding data may not be available, use price-based proxy: z-score of 20-bar returns
combined with volume spikes to detect extreme sentiment (contrarian entry).

Strategy components:
1. 4h HMA(21) = primary trend filter (aligned via mtf_data helper)
2. 12h HMA(21) = higher timeframe bias (aligned via mtf_data helper)
3. Choppiness Index(14) = regime detection (>55 range, <45 trend)
4. Z-score(20) of returns = funding proxy (extreme = contrarian opportunity)
5. Volume ratio(7) = confirmation filter (>1.5 = valid signal)
6. RSI(14) = entry timing within regime
7. ATR(14) trailing stop = 2.5x for risk management

Position sizing: 0.25 base, 0.15 reduced in transition zones
Stoploss: 2.5 * ATR trailing
Timeframe: 1h (REQUIRED for this experiment)

Why this might work:
- 1h captures more opportunities than 4h while avoiding 15m/30m noise
- Z-score contrarian entries work well in 2025 bear/range market
- Volume confirmation filters false breakouts
- Dual HTF (4h + 12h) provides robust trend filter
- Regime-adaptive: mean reversion in ranges, trend-follow in trends

Learning from failures:
- #237 (1h KAMA + ADX): Sharpe=-0.474 - ADX too laggy, no contrarian logic
- #241 (15m trend): Sharpe=-3.471 - too noisy, wrong TF
- Pure trend strategies fail in 2025 bear market
- Need contrarian entries at extremes
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_zscore_volume_4h_12h_hma_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 55 = Range market
    CHOP < 45 = Trend market
    """
    n = len(close)
    chop = np.full(n, 50.0)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_zscore(close, period=20):
    """Calculate z-score of returns (funding rate proxy for contrarian signals)."""
    close_s = pd.Series(close)
    returns = close_s.pct_change()
    rolling_mean = returns.rolling(window=period, min_periods=period).mean()
    rolling_std = returns.rolling(window=period, min_periods=period).std()
    zscore = (returns - rolling_mean) / (rolling_std + 1e-10)
    return zscore.values

def calculate_volume_ratio(volume, period=7):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / (vol_avg.values + 1e-10)
    return vol_ratio

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    chop = calculate_choppiness(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(volume, 7)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_REDUCED = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_close = 0.0
    lowest_close = 999999.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(zscore[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = primary trend filter
        # 12h HMA = higher timeframe confirmation
        bull_4h = close[i] > hma_4h_aligned[i]
        bear_4h = close[i] < hma_4h_aligned[i]
        bull_12h = close[i] > hma_12h_aligned[i]
        bear_12h = close[i] < hma_12h_aligned[i]
        
        # Strong bias when both HTF agree
        strong_bull = bull_4h and bull_12h
        strong_bear = bear_4h and bear_12h
        
        # === REGIME DETECTION ===
        is_range = chop[i] > 55
        is_trend = chop[i] < 45
        is_transition = not is_range and not is_trend
        
        # === CONTRARIAN SIGNALS (Z-score extreme) ===
        # Z-score < -2 = extreme selling (contrarian long)
        # Z-score > +2 = extreme buying (contrarian short)
        zscore_extreme_long = zscore[i] < -1.8
        zscore_extreme_short = zscore[i] > 1.8
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.3
        
        # === MEAN REVERSION SIGNALS (Range Regime) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        price_at_bb_lower = close[i] < bb_lower[i]
        price_at_bb_upper = close[i] > bb_upper[i]
        
        # === TREND FOLLOWING SIGNALS (Trend Regime) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        hma_slope_bull = hma_21[i] > hma_21[i-10] if i >= 10 else False
        hma_slope_bear = hma_21[i] < hma_21[i-10] if i >= 10 else False
        
        new_signal = 0.0
        
        # === RANGE REGIME: MEAN REVERSION + CONTRARIAN ===
        if is_range:
            # Long: RSI oversold + BB lower + zscore extreme OR volume spike
            if rsi_oversold and price_at_bb_lower:
                if zscore_extreme_long or (volume_confirmed and not strong_bear):
                    new_signal = SIZE_BASE
            
            # Short: RSI overbought + BB upper + zscore extreme OR volume spike
            if rsi_overbought and price_at_bb_upper:
                if zscore_extreme_short or (volume_confirmed and not strong_bull):
                    new_signal = -SIZE_BASE
        
        # === TREND REGIME: TREND FOLLOWING WITH PULLBACK ===
        elif is_trend:
            # Long: strong bull HTF + HMA bullish + pullback (RSI 40-55)
            if strong_bull and hma_bullish and hma_slope_bull:
                if 40 < rsi_14[i] < 55:  # Pullback entry
                    new_signal = SIZE_BASE
                elif rsi_14[i] > 55 and volume_confirmed:  # Momentum continuation
                    new_signal = SIZE_REDUCED
            
            # Short: strong bear HTF + HMA bearish + pullback (RSI 45-60)
            if strong_bear and hma_bearish and hma_slope_bear:
                if 45 < rsi_14[i] < 60:  # Pullback entry
                    new_signal = -SIZE_BASE
                elif rsi_14[i] < 45 and volume_confirmed:  # Momentum continuation
                    new_signal = -SIZE_REDUCED
        
        # === TRANSITION REGIME: ONLY CONTRARIAN AT EXTREMES ===
        elif is_transition:
            # Only enter on extreme zscore + volume confirmation
            if zscore_extreme_long and volume_confirmed and rsi_oversold:
                new_signal = SIZE_REDUCED
            if zscore_extreme_short and volume_confirmed and rsi_overbought:
                new_signal = -SIZE_REDUCED
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr:
                    new_signal = SIZE_REDUCED  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr:
                    new_signal = -SIZE_REDUCED  # Take partial profit
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 999999.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 999999.0
        else:
            # Exiting position
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 999999.0
        
        signals[i] = new_signal
    
    return signals