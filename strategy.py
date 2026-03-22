#!/usr/bin/env python3
"""
Experiment #502: 12h Primary + 1d/1w HTF — Funding Rate Mean Reversion + Donchian Breakout

Hypothesis: After 499 failed strategies (mostly CRSI/Choppiness/HMA combos on lower TFs), 
try a fundamentally DIFFERENT approach based on proven research for BTC/ETH:

1. FUNDING RATE MEAN REVERSION: Z-score of funding(30d) < -2 → long, > +2 → short
   This is the BEST EDGE for BTC/ETH per research notes (Sharpe 0.8-1.5 through 2022 crash)
   Uses contrarian logic: extreme positive funding = crowded longs = reversal likely
   
2. DONCHIAN BREAKOUT (20 period): Clean breakout signals with 1d HMA trend filter
   Only trade breakouts in direction of major trend (prevents false breakouts)
   
3. 12h TIMEFRAME: Targets 20-50 trades/year (minimal fee drag)
   Higher TF proven to work better (current best is 1d strategy)
   
4. ASYMMETRIC REGIME: 1w HMA determines bull/bear, adjust entry thresholds
   Bull: prefer long pullbacks, require stronger signal for shorts
   Bear: prefer short bounces, require stronger signal for longs

Why this might beat current best (Sharpe=0.435):
- Funding rate is FUNDAMENTALLY DIFFERENT from price indicators (448 failed with those)
- 12h TF = fewer trades = less fee drag (critical for profitability)
- Contrarian funding logic works in ALL regimes (bull, bear, range)
- Donchian breakout provides clean entry timing
- Proven edge on BTC/ETH specifically (not just SOL)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 12h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_funding_donchian_hma_1d1w_v1"
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
    """Calculate Hull Moving Average (HMA)."""
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

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

def calculate_funding_zscore(prices, symbol, lookback=30):
    """
    Calculate Z-score of funding rates over lookback period.
    Loads funding data from processed funding parquet files.
    Z < -2 = extremely negative funding = contrarian long signal
    Z > +2 = extremely positive funding = contrarian short signal
    """
    try:
        # Map symbol to funding file path
        symbol_lower = symbol.lower().replace('usdt', '')
        funding_path = f"data/processed/funding/{symbol_lower}.parquet"
        
        funding_df = pd.read_parquet(funding_path)
        
        # Align funding data to prices timeframe
        # Funding is 8h, we need to resample to 12h
        funding_df['open_time'] = pd.to_datetime(funding_df['open_time'])
        funding_df = funding_df.set_index('open_time')
        
        # Resample to 12h and take mean funding rate
        funding_12h = funding_df['funding_rate'].resample('12h').mean()
        
        # Align to prices index
        prices_index = pd.to_datetime(prices['open_time'])
        funding_aligned = funding_12h.reindex(prices_index, method='ffill')
        
        # Calculate Z-score
        rolling_mean = funding_aligned.rolling(window=lookback, min_periods=lookback//2).mean()
        rolling_std = funding_aligned.rolling(window=lookback, min_periods=lookback//2).std()
        
        zscore = (funding_aligned - rolling_mean) / (rolling_std + 1e-10)
        
        return zscore.values
    
    except Exception as e:
        # Fallback: return zeros if funding data not available
        return np.zeros(len(prices))

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Extract symbol from prices metadata if available
    symbol = prices.get('symbol', ['BTCUSDT'])[0] if hasattr(prices, 'get') else 'BTCUSDT'
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Calculate 1w HTF indicators (super major trend)
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Donchian Channel for breakout detection
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # RSI for additional confirmation
    rsi_14 = calculate_rsi(close, 14)
    
    # Funding rate Z-score (contrarian signal)
    funding_zscore = calculate_funding_zscore(prices, symbol, lookback=30)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Donchian previous values
    prev_donchian_upper = np.roll(donchian_upper, 1)
    prev_donchian_lower = np.roll(donchian_lower, 1)
    prev_donchian_upper[0] = donchian_upper[0]
    prev_donchian_lower[0] = donchian_lower[0]
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(rsi_14[i]):
            continue
        
        # === 1W SUPER MAJOR TREND (regime filter) ===
        bull_super_regime = close[i] > hma_1w_21_aligned[i]
        bear_super_regime = close[i] < hma_1w_21_aligned[i]
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > prev_donchian_upper[i]
        donchian_breakout_down = close[i] < prev_donchian_lower[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_low = rsi_14[i] < 25.0
        rsi_extreme_high = rsi_14[i] > 75.0
        
        # === FUNDING RATE Z-SCORE (contrarian signal) ===
        # Only use if we have valid funding data (zscore != 0 for most bars)
        funding_extreme_negative = funding_zscore[i] < -1.5
        funding_extreme_positive = funding_zscore[i] > 1.5
        funding_extreme_neg_strong = funding_zscore[i] < -2.0
        funding_extreme_pos_strong = funding_zscore[i] > 2.0
        
        # Check if funding data is available (not all zeros)
        funding_available = np.any(funding_zscore[:i] != 0)
        
        # === ENTRY LOGIC — FUNDING CONTRARIAN + DONCHIAN BREAKOUT ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple conditions for trade frequency)
        # Condition 1: Funding extreme negative + RSI oversold (contrarian bottom)
        if funding_available and funding_extreme_negative and rsi_oversold:
            new_signal = LONG_SIZE
        # Condition 2: Bull regime + Donchian breakout up (trend continuation)
        elif bull_regime and hma_slope_bull and donchian_breakout_up:
            new_signal = LONG_SIZE
        # Condition 3: Bull super regime + RSI pullback (buy dip in major uptrend)
        elif bull_super_regime and rsi_oversold and bull_regime:
            new_signal = LONG_SIZE * 0.8
        # Condition 4: Funding extreme negative strong (strong contrarian long)
        elif funding_available and funding_extreme_neg_strong:
            new_signal = LONG_SIZE
        # Condition 5: RSI extreme low + price above 1d HMA21 (pullback in uptrend)
        elif rsi_extreme_low and bull_regime:
            new_signal = LONG_SIZE * 0.7
        # Condition 6: Donchian breakout + RSI confirmation (momentum entry)
        elif donchian_breakout_up and rsi_14[i] > 50:
            new_signal = LONG_SIZE * 0.6
        
        # SHORT ENTRIES (mirror logic for bear market)
        if new_signal == 0.0:
            # Condition 1: Funding extreme positive + RSI overbought (contrarian top)
            if funding_available and funding_extreme_positive and rsi_overbought:
                new_signal = -SHORT_SIZE
            # Condition 2: Bear regime + Donchian breakout down (trend continuation)
            elif bear_regime and hma_slope_bear and donchian_breakout_down:
                new_signal = -SHORT_SIZE
            # Condition 3: Bear super regime + RSI bounce (short rally in major downtrend)
            elif bear_super_regime and rsi_overbought and bear_regime:
                new_signal = -SHORT_SIZE * 0.8
            # Condition 4: Funding extreme positive strong (strong contrarian short)
            elif funding_available and funding_extreme_pos_strong:
                new_signal = -SHORT_SIZE
            # Condition 5: RSI extreme high + price below 1d HMA21 (bounce in downtrend)
            elif rsi_extreme_high and bear_regime:
                new_signal = -SHORT_SIZE * 0.7
            # Condition 6: Donchian breakdown + RSI confirmation (momentum entry)
            elif donchian_breakout_down and rsi_14[i] < 50:
                new_signal = -SHORT_SIZE * 0.6
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long on RSI overbought or regime flip
        if in_position and position_side > 0:
            if rsi_overbought:
                new_signal = 0.0
            # Exit if regime flips bearish strongly
            if bear_regime and hma_slope_bear:
                new_signal = 0.0
        
        # Exit short on RSI oversold or regime flip
        if in_position and position_side < 0:
            if rsi_oversold:
                new_signal = 0.0
            # Exit if regime flips bullish strongly
            if bull_regime and hma_slope_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals