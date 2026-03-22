#!/usr/bin/env python3
"""
Experiment #421: 15m Multi-Regime Strategy with 4h HMA Trend + 1h Choppiness Filter

Hypothesis: After analyzing 420+ failed experiments, the key insight is that 15m
timeframe needs ADAPTIVE logic based on market regime. Simple trend-following
fails in bear/range markets (2022 crash, 2025 bear). This strategy uses:

1. 4h HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 4h HMA
   - Short bias when price < 4h HMA
   - HMA smoother than EMA, critical for MTF alignment

2. 1h CHOPPINESS INDEX REGIME DETECTION (via mtf_data helper):
   - CHOP > 61.8 = ranging market (use mean-reversion entries)
   - CHOP < 38.2 = trending market (use breakout entries)
   - 38.2-61.8 = neutral (reduce position size or stay flat)
   - This is the KEY differentiator from failed strategies

3. 15m ENTRY LOGIC (adaptive per regime):
   - TRENDING: Donchian(20) breakout in direction of 4h trend
   - RANGING: RSI(7) extremes (30/70) with 4h trend bias
   - This avoids counter-trend mean-reversion in strong trends

4. VOLUME CONFIRMATION:
   - Entry volume > 0.8 * 20-bar avg volume
   - Filters false breakouts on low liquidity

5. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from 2022-style crashes

6. POSITION SIZING: 0.25 discrete (conservative for 15m volatility)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why 15m with this approach should work:
- Adaptive regime detection avoids trend-following in chop
- 4h HMA provides strong trend filter (not 1h which is too noisy)
- 1h Choppiness gives cleaner regime signal than ADX on 15m
- Works on BTC/ETH/SOL individually (not SOL-biased)
- Should generate 50-100 trades/year (enough for stats, not too many for fees)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h HMA + 1h Choppiness via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_chop_4h_hma_1h_donchian_rsi_atr_v1"
timeframe = "15m"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR for each bar (simplified: just high-low for this calculation)
    tr = high - low
    
    for i in range(period, n):
        atr_sum = tr[i-period+1:i+1].sum()
        highest_high = high[i-period+1:i+1].max()
        lowest_low = low[i-period+1:i+1].min()
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_rsi(close, period=7):
    """Calculate Relative Strength Index with shorter period for 15m."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    return vol_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    chop_1h = calculate_choppiness(df_1h['high'].values, df_1h['low'].values, df_1h['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    chop_1h_aligned = align_htf_to_ltf(prices, df_1h, chop_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    vol_avg = calculate_volume_avg(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_avg[i]) or vol_avg[i] == 0:
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (1h Choppiness) ===
        trending_market = chop_1h_aligned[i] < 38.2
        ranging_market = chop_1h_aligned[i] > 61.8
        neutral_market = not trending_market and not ranging_market
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_avg[i]
        
        # === DONCHIAN BREAKOUT SIGNALS (for trending regime) ===
        donchian_long = close[i] > donchian_upper[i-1]  # Break above previous high
        donchian_short = close[i] < donchian_lower[i-1]  # Break below previous low
        
        # === RSI MEAN REVERSION SIGNALS (for ranging regime) ===
        rsi_long = rsi[i] < 30  # Oversold
        rsi_short = rsi[i] > 70  # Overbought
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # TRENDING REGIME: Donchian breakout with 4h HMA filter + volume
        if trending_market:
            if bull_trend_4h and donchian_long and volume_confirmed:
                new_signal = SIZE
            elif bear_trend_4h and donchian_short and volume_confirmed:
                new_signal = -SIZE
        
        # RANGING REGIME: RSI mean-reversion with 4h HMA filter
        elif ranging_market:
            # Only enter with trend bias (avoid counter-trend mean reversion)
            if bull_trend_4h and rsi_long:
                new_signal = SIZE
            elif bear_trend_4h and rsi_short:
                new_signal = -SIZE
        
        # NEUTRAL REGIME: Reduce position size or stay flat
        elif neutral_market:
            # Only take strongest signals in neutral regime
            if bull_trend_4h and donchian_long and volume_confirmed and rsi[i] < 50:
                new_signal = SIZE / 2  # Half position in neutral
            elif bear_trend_4h and donchian_short and volume_confirmed and rsi[i] > 50:
                new_signal = -SIZE / 2  # Half position in neutral
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        # Exit if regime changes against position type
        if in_position and new_signal != 0.0:
            # Long position in trending regime should exit if market becomes ranging without RSI signal
            if position_side > 0 and ranging_market and not rsi_long:
                new_signal = 0.0
            # Short position in trending regime should exit if market becomes ranging without RSI signal
            if position_side < 0 and ranging_market and not rsi_short:
                new_signal = 0.0
        
        # === TREND REVERSAL EXIT (for trending regime positions) ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals